# MFSDP v2 + combined/fine-grained 1F1B: schedule-delimited gradient completion

Design note for the fix on branch `fix/mfsdp-v2-mtp-support`, based on
`fc85a8cd46de0f07253c9bd5fec51df3ff966b4d`.

## 1. The invariant being restored

> In the combined/fine-grained 1F1B path, an `FsdpModule` issues its reshard and its
> gradient reduce **exactly once per backward window**, at the point where the schedule
> declares that module's backward over, and only after consulting the actual gradient
> state of the module's trainable parameters.

Two weaker invariants were being relied on before:

* *Old (combined path):* "the reduce is issued when the `V`-th post-accumulate-grad
  callback arrives", where `V` is the number of distinct trainable `FsdpParameter`s
  (`Countdown`, `experimental/countdown.py`, armed at `experimental/module.py`).
* *True precondition of that rule:* one callback per trainable parameter per backward
  window. That holds only when the module's whole forward+backward runs in one autograd
  graph task, i.e. the automatic path (`register_hooks=True`).

The combined schedule violates the precondition structurally: it runs **one
`run_backward` per schedule node on detached node inputs**
(`megatron/core/pipeline_parallel/utils.py`), so a parameter's callback count per window
is its number of *(parameter, consuming-node)* pairs, not 1. In the MTP case the shared
embedding is consumed by the `PreProcessNode` **and** by the MTP pre-dispatch node, so the
root module over-fires by one per MTP depth and completes one callback early — while
`decoder.final_layernorm.weight`, applied only by the last decoder layer's combine node,
still has no gradient. The premature reduce then raises
`Missing gradient for FSDP parameter ('module.decoder.final_layernorm.weight',)`. The
mirror case (a parameter used by no node in the window) under-fires, leaves unspent
credit in the counter, and silently drops the reduce-scatter.

## 2. The smallest change that restores it

Four edits, all inside the existing structure:

1. **`GradientReadiness`** (`experimental/countdown.py`): records *which* parameters
   fired, keyed by index in the module's trainable order, instead of *how many*
   callbacks arrived. A repeated callback is idempotent; `missing()` is the ground-truth
   set of unused parameters at window close; `close()` re-arms for the next window.
2. **`FsdpModule`** (`experimental/module.py`):
   * `register_post_backward_hook(..., schedule_driven=True)` registers readiness marks
     and *holds* the reduce hook instead of calling it from the callback;
   * `finalize_scheduled_backward()` is the single reduce trigger: it closes the window,
     warns once with the FQNs of any unused parameters, then calls the held hook;
   * `assert_scheduled_backward_closed()` is the loud end-of-chunk invariant;
   * `_reduce_gradient_groups()` passes `require_all_grads=self._grad_readiness is None`,
     so only a schedule-delimited module may zero-fill.
3. **`FsdpParameterGroup`** (`experimental/parameter_group.py`): the packed partial-grad
   buffer is laid out from the *parameter group* (one slot per parameter, dtype/device
   from a real gradient when present, `parameter.shape` otherwise) and
   `copy_gradients_to_partial_buffer` zero-fills an absent slot. The strict
   `Missing gradient` raise is kept as the default, so the automatic path is unchanged.
4. **Wiring**: `register_combined_1f1b_hooks()` asks for `schedule_driven=True`;
   `model_chunk_schedule_plan.py` stops discarding the backward edge
   (`set_fsdp_reshard_hooks(reshard_fsdp_module, finalize_fsdp_backward)`), and closes the
   chunk root — which owns the embedding, the output layer, the decoder final norm and the
   MTP pre/post-processing weights — at the chunk's last backward operation, the
   pre-process node.

`finalize_fsdp_backward(module)` closes the unit *and every FSDP unit nested inside it*:
an expert parameter group is its own `FsdpModule` inside a MoE layer, and it finishes with
the layer that contains it. That is what makes the pre-existing EP-overlap path
(`deepseek_proxy_mfsdp_v2_ep2`, `mtp_num_layers=0`) close correctly rather than trip the
end-of-chunk assertion.

## 3. Why here, not in the scheduler or the optimizer

* **Not in the scheduler.** The scheduler knows *when* a backward ends; it does not know
  *which* parameter belongs to which `FsdpModule`, nor whether a gradient is present, nor
  the packed-buffer layout. Pushing the decision there would duplicate `FsdpModule`'s
  ownership model into model-chunk code. Instead the scheduler only *declares* the edge —
  which it already did, and threw away with `lambda _: None`.
* **Not in the optimizer.** The optimizer runs after the whole step and cannot repair a
  reduce that was issued early (grads already packed and cleared) or never issued (the
  contribution is lost). Also, only `FsdpModule` can observe the per-parameter callbacks.
* **Where it lives** is exactly where the false inference lived: the completion signal in
  `countdown.py`, its use in `module.py`, and the ground truth in `parameter_group.py`.

## 4. `mtp_num_layers == 0` blast radius

* **Automatic path (`register_hooks=True`, i.e. `overlap_moe_expert_parallel_comm=False`),
  any `mtp_num_layers`:** untouched. `_grad_readiness` stays `None`, so
  `register_post_backward_hook` takes the original countdown branch, the countdown still
  triggers the hook, `_reduce_gradient_groups` keeps `require_all_grads=True`, and the
  `Missing gradient` raise is byte-for-byte the same. The extra methods are no-ops.
* **Combined path (`register_hooks=False`, `overlap_moe_expert_parallel_comm=True`) with
  `mtp_num_layers=0`: this path DOES change.** Its completion signal moves from the
  countdown to the schedule edge. That is intentional — the same segmented backward that
  MTP exposes also runs with MTP disabled (the prior reproduction on
  `origin/repro/pr-a-fsdp-v2-countdown-cycle` has no MTP), and the schedule edge is the
  correct signal there too. Functionally the no-MTP path should be equivalent: each layer
  unit's edge fires right after its own last backward node, which is where the countdown
  completed anyway, and the root's window now closes at the chunk's last backward
  operation. The deltas to validate on GPU are (a) `deepseek_proxy_mfsdp_v2_ep2` losses
  and (b) that the end-of-chunk assertion does not fire.
* **`mtp_num_layers=0` + no EP overlap** is the regression gate for the countdown.

## 5. Hazards handled explicitly

* **Rank-uniformity (collective safety).** The zero-fill is decided locally but never
  changes the collective: the buffer has one slot per parameter on every rank and
  `_reduce_gradient_groups` still issues exactly one reduce-scatter per parameter group
  per window. Nothing skips, and the missing/not-missing branch happens before the
  collective, so a rank-asymmetric miss cannot desynchronise a collective and hang.
  Scheduling is identical across DP ranks (same model, same chunk), so the *set* of
  modules and windows is uniform too.
* **Buffer dtype/shape stability.** `allocate_partial_grad_buffer` derives the layout
  from every parameter in the group, with dtype/device from the first *present* gradient
  and `main_grad.dtype/device` as the all-absent fallback; `parameter.shape` is used only
  for a slot with no gradient (where it equals the gradient shape anyway). With every
  gradient present the buffer is identical to before.
* **Silence vs. loudness.** An unused parameter is a legitimate zero contribution, so it
  is zero-filled and *reported once per module* at WARNING with its FQNs. An open window
  at the end of a chunk is a schedule bug and raises.

## 6. `mtp_num_layers=0` / defect-2 interaction (Muon + MXFP8 + MTP deadlock)

Defect 2 is **not** fixed here. Its two surviving hypotheses (H1: the skew is at the
embedding communicator opened by `finalize_model_grads._allreduce_word_embedding_grads` /
`_allreduce_embedding_grad`; H2: it is at a data-parallel/bucket communicator) cannot be
separated without the GPU op-count experiment that a separate subtask is running, so any
fix would be speculative. The constraint is recorded here: **if H1 is confirmed, the
embedding all-reduce's participation predicate must become globally uniform** — a tiny
`int32` all-reduce over the embedding group, then an identical branch on every rank,
following the existing precedent at `megatron/core/optimizer/__init__.py:323-349`.

Interaction with this fix, stated explicitly because the over-fired root parameter in the
MTP case *is* the shared embedding:

* This change does not add, remove or condition any collective. It only moves *when* the
  root's reshard+reduce-scatter happens: previously wherever the countdown crossed zero
  during the chunk backward, now deterministically at the chunk's last backward operation.
  The reduce-scatter is per-`FsdpModule` and per-DP-group, so it is not the PP-spanning
  embedding all-reduce and is not a new source of entry skew.
* It does *not* remove the entry skew that H1 blames: the embedding all-reduce in
  `finalize_model_grads` is gated by which chunks hold the embedding/output layer, and MTP
  adds the MTP layer's forward/backward to only the last VP chunk of the last PP stage.
  That gating is untouched.
* It does remove one contributor to *timing* noise on the last stage: the root's
  reduce-scatter no longer fires at an MTP-depth-dependent point. If the op-count
  experiment shows the hang is sensitive to ordering around that point, this fix changes
  the observed schedule and the experiment should be re-read against the fixed tree.
* Zero-filling never changes the participation set, so it cannot turn an H1 skew into a
  hang on its own.

## 7. Residual unsupported / unverified cases

1. **Delayed wgrad (`skip_backward_post_hook`, TE `backward_dw`).** `TransformerLayerNode`
   collects `post_wgrad_grad_acc_hooks` once per iteration and only for parameters whose
   `grad is not None` at that moment, so a delayed-wgrad parameter can be dropped from the
   hook list. The schedule edge fires after `backward_dw` (so the ordering is right), but
   a parameter whose gradient never materialises in the window will now be zero-filled
   rather than flagged as a bug. Untested on GPU.
2. **`--mtp-use-repeated-layer` with `mtp_num_layers > 1`.** One
   `MultiTokenPredictionLayer` is applied `mtp_num_layers` times while the combined plan
   list has length `len(model.mtp.layers) == 1`. The new signal makes the *completion*
   phase-stable and idempotent, but it does not reconcile the plan count with the loss
   count; a repeated MTP layer whose gradients arrive outside any plan edge is still
   structurally mismatched.
3. **VPP chunk reuse.** Each `FsdpModule` now has a per-module window closed by its own
   chunk's edge, which should remove the cross-chunk phase drift of class D, but the
   interleaved-pipelining path was not exercised here.
4. **A parameter that fires only after its module's edge.** That re-opens the window and
   trips the end-of-chunk assertion (loud, not silent). No such ordering is known for GPT
   or MTP, but it is the case to watch in a GPU run.
5. **`GradientReadiness` vs. the countdown for the automatic path** is a deliberate
   non-migration; if the automatic path later gains a segmented backward, it needs the
   same treatment.
6. Everything in §4 and the zero-fill path is **unverified on GPU**: the unit tests
   exercise `GradientReadiness` and the `FsdpModule` completion policy, not a real
   training step. A GPU run must confirm `deepseek_proxy_mfsdp_v2_ep2` (no MTP) parity and
   an MTP (`mtp_num_layers=1`, then `--mtp-use-repeated-layer`) step that previously
   raised `Missing gradient for FSDP parameter`.
