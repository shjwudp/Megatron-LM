# MFSDP v2 gradient completion by schedule-declared multiplicity

Status: implementation for review. CPU-verified; the GPU unit tests are the
remaining gate. Branch: `feat/mfsdp-v2-grad-multiplicity`, stacked on
`fix/mfsdp-v2-mtp-support` (`ac7092edd`).

## 1. The problem

`FsdpModule` inferred "this unit's backward is finished" from a fire **count**
whose period was the number of trainable parameters it owns. That is valid only
when each parameter contributes exactly once per window, and the
combined/fine-grained 1F1B backward does not guarantee it: it runs one autograd
GraphTask per schedule node over detached node inputs, so a parameter contributes
once per **consuming node**.

Two verified premises (CPU, `torch.autograd`, see
`multiplicity_readiness_study.py`):

1. a parameter's `.grad` presence is **monotone** across GraphTasks -- once set it
   never returns to `None` -- so presence cannot distinguish "done" from "still
   accumulating" and counting is required;
2. fires per parameter equal the number of GraphTasks that consume it (1 graph ->
   1 fire, 3 graphs -> 3 fires), so a per-parameter expected count
   ("multiplicity") is well defined and observable.

The shipped fix on `fix/mfsdp-v2-mtp-support` sidesteps the defect with
idempotent marks plus a schedule-declared edge (see
`mtp-countdown-fix-design.md`). It is correct, but it loses information: an
idempotent mark cannot distinguish "nothing left to do" from "something never
ran", and an edge that arrives **before** the last application leaks a mark into
the next window. Both of those are now observable.

## 2. The design

`MultiplicityReadiness` (`experimental/countdown.py`) is exact per-parameter
accounting against a schedule-declared multiplicity:

```python
complete   <=>  every parameter's count == its declared multiplicity
under-fire ->  count <  multiplicity   # reported at the close edge
over-fire  ->  count >  multiplicity   # recorded at the offending mark
```

One hook set now serves both backward modes; only the **provider** of the
expected counts differs:

* **automatic path** (`register_hooks=True`, i.e.
  `overlap_moe_expert_parallel_comm=False`): every trainable parameter is declared
  1. The accounting therefore degenerates to exactly the previous count-based
  semantics -- the window closes on the callback that completes the last
  parameter, and `_reduce_gradient_groups` keeps its strict
  `require_all_grads=True` check.
* **combined/fine-grained 1F1B**: the schedule declares the real counts, the
  window stays open until the schedule's end-of-backward edge, and a short
  parameter is zero-filled instead of aborting the reduce.

The `schedule_driven: bool` parameter is gone; `register_post_backward_hook` takes
`grad_multiplicity: bool`, which selects only whether the declaration is consulted
and whether the hook is held for the schedule edge. `GradientReadiness` and
`Countdown` are no longer used for completion anywhere in production; they stay in
the module because the mechanism tests and the design study exercise them, and
they are the evidence for why they were replaced.

## 3. The safety argument: why the asymmetry matters

Declaring too **few** is the original defect: the window closes before a surplus
contribution arrives, and that contribution is then charged to the next window.
Declaring too **many** merely waits. Therefore:

* **over-fire is reported loudly**, once per unit (`_warned_over_fire`), naming the
  parameter FQNs, the observed count and the declared count:
  `MFSDP module ... observed MORE gradient contributions than its declared
  multiplicity ... the surplus is charged to a later window`. It means the
  multiplicity was under-declared and the window already closed early.
* **under-fire at the close edge is reported loudly**, once per unit
  (`_warned_missing_grads`), naming each parameter, its observed count and its
  declared count. The two explanations are an over-declaration and a schedule node
  that never ran (the silent dropped reduce-scatter case).
* a module whose window never opened is still a no-op, and a window in which every
  parameter hit its declaration reports nothing (so the automatic path stays
  quiet).
* `assert_scheduled_backward_closed` still raises at the end of a chunk, now with
  both the shortfall and the over-fire in the message, so a late contribution is a
  loud surplus rather than a silent leaked mark.

The reporting happens inside `FsdpModule._report_grad_window`, called from
`_close_grad_window` **before** `MultiplicityReadiness.close()` resets the counts,
which is what makes the counts in the message the ones that describe the window.
`MultiplicityReadiness` itself stays stdlib-only so the tests can load it from
source with no CUDA/TE stack.

## 4. The multiplicity provider

`combined_1f1b_mfsdp_scheduler._unit_grad_multiplicity(unit, shared_weight,
mtp_depth)` walks the unit's trainable parameters (the same key space the marks
use) and declares non-1 counts for the two cases that actually occur:

* **shared embedding consumed by the pre-process node and every MTP pre-dispatch
  node**: `1 + mtp_depth`, where `mtp_depth` is the number of MTP layers this
  pipeline stage actually runs (`_active_mtp_layers`, derived with the same
  `get_mtp_layer_offset` the schedule uses). Evidence:
  `megatron/core/models/common/fine_grained_callables.py:74-77` calls
  `layer._get_embeddings(..., embedding=node.chunk_state.model.embedding, ...)`
  from `submodule_mtp_pre_dispatch_forward`, i.e. the MTP pre-dispatch node
  explicitly uses `model.embedding`.
* **tied embedding/output weight** (`share_embeddings_and_output_weights`, found by
  identity via `shared_embedding_or_output_weight()`): add `1 + mtp_depth`, the
  chunk's post-process output projection plus one projection per MTP head
  (`process_mtp_loss` calls `output_layer` once per MTP depth).
* everything else: 1 (absent from the mapping).

Every `FsdpModule` in the chunk is given its own declaration, so the arithmetic
does not depend on which unit happens to own the shared weight. `mtp_depth == 0`
makes the embedding rule `1` and the tied-weight rule `2`, which is exactly the
no-MTP combined case.

### What is NOT covered

* **MoE expert units and any other nested unit.** Each nested unit is closed by its
  own layer edge and every one of its parameters is consumed by exactly one node of
  that layer, so its declaration is empty. If that is ever untrue the unit
  over-fires loudly.
* **`--mtp-use-repeated-layer` with `mtp_num_layers > 1`.** One
  `MultiTokenPredictionLayer` object is applied several times while
  `len(model.mtp.layers) == 1`, so `_active_mtp_layers` undercounts the embedding
  consumers. That direction is the loud one (over-fire), not the silent one.
* **A parameter consumed outside the plan's node set.** Any consumer the
  enumeration does not know about over-fires; any declared consumer that never runs
  under-fires. Neither is silent.
* **Delayed wgrad (TE `backward_dw`) per-parameter counts.** The delayed path is
  routed through the same accounting (see below), but the old caveat stands: the
  node collects `post_wgrad_grad_acc_hooks` only for parameters whose `grad is not
  None` at that moment, so a delayed-wgrad parameter can still come up short. It is
  now reported instead of silently zero-filled.
* **Parity with the previous behaviour on real training.** Not verifiable on CPU.

## 5. One clock for TE delayed wgrad

Parameters with `skip_backward_post_hook` do not fire
`register_post_accumulate_grad_hook`; they are materialized by `backward_dw()` and
routed through `parameter_module.register_wgrad_accumulation_and_reduce_hooks`.
Both registrations now call the same `FsdpModule._record_grad_contribution(index)`,
so the multiplicity accounting holds for both. The explicit error for tied
parameters with delayed wgrad (`len(fsdp_parameter.fqns) > 1` is unsupported) is
unchanged.

## 6. Invariants preserved

* `mtp_num_layers == 0` behaviour is unchanged for the automatic path, bit for bit:
  expected counts are all 1, the hook still fires on the last parameter callback,
  and the strict missing-gradient raise is untouched.
* The zero-fill decision never changes the collective: one slot per parameter on
  every rank, one reduce-scatter per parameter group per window.
* No existing assertion was weakened. `assert_scheduled_backward_closed` gained the
  over-fire detail; its raise is the same failure mode.
* `countdown.py` remains dependency-free and loadable from source by path, which is
  what keeps the CPU mechanism tests runnable anywhere.

## 7. Evidence

* `tests/unit_tests/distributed/mfsdp_v2/test_mfsdp_v2_grad_readiness.py` extended
  with `TestMultiplicityReadiness` (exact-once, shared consumer, under-fire,
  over-fire before and at the close edge, unknown key, reset, five identical
  windows, declaration-too-small) and with `FsdpModule`-level tests for the loud
  under-fire and over-fire reports and the end-of-chunk raise. Existing tests were
  kept and adapted to the new signal, not deleted.
* `multiplicity_readiness_study.py` now loads `MultiplicityReadiness` from
  `countdown.py` itself and reproduces the original comparison table verdicts.
* `tools/autoformat.sh` (CHECK_ONLY) passes on the branch file set.
