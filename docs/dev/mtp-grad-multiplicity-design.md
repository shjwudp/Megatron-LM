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

The space is small because the combined path is only reachable with
`overlap_moe_expert_parallel_comm`, and that configuration asserts
`mtp_num_layers in (None, 0, 1)` at config-construction time:

```text
megatron/core/transformer/transformer_config.py:3316-3320
    assert self.mtp_num_layers in (None, 0, 1), \
        'MTP supports at most one layer when enabling overlap_moe_expert_parallel_comm.'
```

So the whole declaration is enumerable. `_active_mtp_layers` reads the model's own
MTP block -- 0 or 1, because a pipeline stage that does not own the block reports 0
-- and raises if it ever sees more, and
`_unit_grad_multiplicity(unit, shared_weight, mtp_depth)` walks the unit's trainable
parameters (the same key space the marks use) and declares:

| parameter | `mtp_depth=0` | `mtp_depth=1` | consumers |
| --- | --- | --- | --- |
| embedding weight | 1 | 2 | `PreProcessNode` + one MTP pre-dispatch node per depth |
| tied embedding/output weight (same object) | 2 | 3 | the above + the post-process output projection |
| untied output weight | 1 (default) | 1 (default) | its own output projection, once |
| everything else | 1 (default) | 1 (default) | one node of this unit |

Evidence for the embedding term:
`megatron/core/models/common/fine_grained_callables.py:74-77` calls
`layer._get_embeddings(..., embedding=node.chunk_state.model.embedding, ...)` from
`submodule_mtp_pre_dispatch_forward`, i.e. the MTP pre-dispatch node explicitly
uses `model.embedding`. The tied-output term is the same weight object being reused
by the chunk's `PostProcessNode` `output_layer` call, matched by identity through
`shared_embedding_or_output_weight()`.

`mtp_num_layers == 0` collapses to all-1s unless the embeddings and output weight
are tied, in which case the tied weight is 2: a tied weight really is consumed by
both the pre-process lookup and the post-process projection, and adding a consumer
that is not there would be an over-declaration.

Every `FsdpModule` in the chunk gets its own declaration, so the arithmetic does not
depend on which unit happens to own the shared weight.

### The scheduler must not import MTP at module scope

`combined_1f1b_mfsdp_scheduler.py` is imported from
`mcore_fsdp_adapter.py`, which `megatron/core/optimizer/__init__.py` pulls in while
`megatron/core/__init__.py` is still executing -- `:8` imports `distributed`, and
`:9` binds `InferenceParams`. A module-level
`from megatron.core.transformer.multi_token_prediction import ...` therefore closes
the cycle

```text
mcore_fsdp_adapter -> combined_1f1b_mfsdp_scheduler -> multi_token_prediction
  -> from megatron.core import InferenceParams   # not bound yet
```

and breaks `import megatron.core` for the entire repository at pytest collection
time. The provider above is written to need nothing from that module: it reads the
MTP depth from the model's own block, so the cycle cannot recur through it.
`tests/unit_tests/distributed/mfsdp_v2/test_mfsdp_v2_scheduler_import.py` pins this
both statically (AST: no module-level import of `multi_token_prediction`, runnable
anywhere) and dynamically (the imports succeed, with `pytest.fail` rather than a
skip marker so it cannot degrade quietly).

### What is NOT covered

* **MoE expert units and any other nested unit.** Each nested unit is closed by its
  own layer edge and every one of its parameters is consumed by exactly one node of
  that layer, so its declaration is empty. If that is ever untrue the unit
  over-fires loudly.
* **`--mtp-use-repeated-layer` with `mtp_num_layers > 1`.** Not reachable in the
  combined path: `transformer_config.py:3316-3320` rejects it before any model is
  built, and at depth 1 the flag is a no-op. There is no depth>1 machinery, and
  `_active_mtp_layers` raises if it ever sees more than one MTP layer. (The
  root-cause study's "class C", where the plan counts `len(model.mtp.layers) == 1`
  while the loss depth is `config.mtp_num_layers`, is therefore unreachable too.)
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
  windows, declaration-too-small), with `FsdpModule`-level tests for the loud
  under-fire and over-fire reports and the end-of-chunk raise, and with
  `TestReportedGradWindowMessage`, which asserts the report text against the real
  `FsdpModule`. Existing tests were kept and adapted to the new signal, not
  deleted.
* `tests/unit_tests/distributed/mfsdp_v2/test_mfsdp_v2_scheduler_import.py` is new:
  it pins the module-scope import ban statically (AST, runnable on any host) and
  asserts the imports succeed, failing rather than skipping.
* `multiplicity_readiness_study.py` now loads `MultiplicityReadiness` from
  `countdown.py` itself and reproduces the original comparison table verdicts.
* `tools/autoformat.sh` (CHECK_ONLY) passes on the branch file set, as does
  `compileall`.
* **h100 GPU unit tests** (`mcore-devtoolkit unit-test`, cw-mtp, 1 node x 8 GPUs,
  `unit-tests` recipe, `run_ci_test.sh`): re-run against the exact pushed tree after
  the verification incident below. On that stack every GPU-stack test runs instead
  of skipping, and the recorded warnings confirm the accounting fires: an under-fire
  report naming `(('unit.param0',), 0, 2)` and an over-fire report naming
  `(('unit.param0',), 2, 1)`. The exact counts for this revision are in the pull
  request body, so this note never states a number that was measured on an earlier
  tree.

### Verification incident (recorded because it changed the outcome)

An earlier revision of this branch was pushed with the cycle described in section 4
still present, and it broke `import megatron.core`. The lazy-import fix and the
provider correction had only ever existed in the working tree: a history rewrite
restored the files from a pre-fix commit, and the "nothing changed" check compared
against a backup that pointed at that same pre-fix commit, so it could not detect
the loss. Compounding it, this host's torch is old enough that `import megatron.core`
fails earlier, inside `mxfp8_tensor.py`, so the local runs never reached the cycle
and the GPU runs that did were not repeated after the rewrite. The lesson is in the
test file above: the guard is a static property of the source, not an environment
probe, so it holds on every host including the ones that cannot import the stack.

### Deliberately not verified here

No GPU training step was run. In particular, the no-MTP combined loss parity and
the MTP step that previously raised `Missing gradient for FSDP parameter
('module.decoder.final_layernorm.weight',)` are not re-measured; the completion
signal is covered at the mechanism level only.
