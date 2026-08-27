# 1F1B EP Overlap — MFSDP v2 Integration Design

This document describes how the experimental Megatron FSDP v2
(`megatron_fsdp.experimental`) integrates with the 1F1B EP-overlap schedule
(`combined_1f1b`). The v1 implementation (`megatron_fsdp`) defines the
schedule-facing contract; this PR implements the same contract for v2.

---

## 0. Background — Why the Overlap Schedule Needs Special FSDP Handling

### Normal FSDP flow

FSDP hooks fire on `TransformerLayer`:

```
TransformerLayer.forward()
  → pre_forward hook:  unshard_parameters()
  → actual compute
  → post_forward hook: reshard_parameters()
backward()
  → pre_backward hook:  unshard_parameters()
  → compute grads
  → post_backward hook: reduce_grad() + reshard_parameters()
```

### EP overlap flow

The combined 1F1B schedule calls **sub-modules directly**, bypassing
`TransformerLayer.forward()`:

```
combined_forward_backward_step()
  → f_layer.attn.forward()     ← no TransformerLayer hook fires
  → b_layer.mlp.backward()     ← no TransformerLayer hook fires
  → f_layer.moe_dispatch.forward()
  → ...
```

Because the schedule invokes sub-modules (`attn`, `mlp`, `moe_dispatch`,
`moe_combine`) directly, FSDP hooks registered on `TransformerLayer` are
**never triggered**. FSDP must expose manual management APIs that the schedule
calls explicitly at the right moments.

### Key flags

| Flag | What it does |
|---|---|
| `--overlap-moe-expert-parallel-comm` | Enables the combined 1F1B schedule and fine-grained FSDP hooks. |
| `--delay-wgrad-compute` | TE layers postpone weight gradient computation until `backward_dw()`. FSDP must defer `reduce_grad()` until after `backward_dw()`. |

### Why `delay_wgrad_compute` matters for FSDP

`post_accumulate_grad_hook` observes execution of the parameter's autograd
`AccumulateGrad` node; it is not a general observer of later writes to
`parameter.grad`. With `delay_wgrad_compute=True`, that node runs during the
main backward before TE's later `backward_dw()` call writes the delayed weight
gradient. `backward_dw()` does not execute `AccumulateGrad` a second time, so
reducing from the hook would miss the delayed contribution. The overlap adapter
therefore disables the generic FSDP forward/backward lifecycle, including its
parameter-completion callbacks, and lets the explicit
`post_backward_release_module()` callback reduce after `backward_dw()`.

---

## 1. Schedule Integration Contract

The schedule in `combined_1f1b.py` has six integration points where it calls
into FSDP. MFSDP v2 satisfies all six.

### 1.1 Discovery — `find_megatron_fsdp(model)`

**File:** `megatron_fsdp/utils.py`

`find_megatron_fsdp()` walks the model wrapper chain and returns either a v1
`MegatronFSDP` wrapper or a `FullyShardedDataParallelV2` adapter wrapping an
`FsdpModule`. Returning the adapter is important because the schedule-facing
callbacks live there, not on the bare `FsdpModule`.

### 1.2 Pre-schedule setup — `_replace_param_with_raw_if_needed()`

**File:** `combined_1f1b.py` → `FullyShardedDataParallelV2._replace_param_with_raw_if_needed()`

The schedule calls this once before running. In v1 it swaps optimizer-facing
DTensor parameters back to raw `nn.Parameter`s. MFSDP v2 always stores raw
tensors backed by `DBuffer` on the module, so **no parameter swap is needed**;
the adapter finalizes the root `FsdpContext` instead, ensuring a child FSDP
unit cannot mistake itself for the root when it executes first:

```python
# In FullyShardedDataParallelV2._setup_1f1b_overlap_interface (mcore_fsdp_adapter.py)
def _replace_param_with_raw_if_needed() -> None:
    self.module.context.ensure_finalized()
```

### 1.3 Root backward-phase setup — `pre_backward()`

**File:** `combined_1f1b.py` → `FsdpModule.pre_backward()`

Called before each combined backward segment. MFSDP v2:

- transitions this module to `Phase.BACKWARD`,
- if root: forks the reduce-scatter stream from the current stream so later
  allocations stay legal under CUDA-graph capture; the schedule passes
  `register_final_callback=False` because it finalizes explicitly,
- unshards parameters (all-gather) and waits for them,
- prefetches the next module in `backward_order` (static-order prefetch).

### 1.4 Per-layer parameter release — `set_fsdp_reshard_hooks()`

**File:** `combined_1f1b.py` → adapter callbacks

The schedule attaches two callables per `TransformerLayerSchedulePlan`:

- **`post_forward_release_module(module)`** — called after the last forward
  node of a layer; reshard only (no gradient reduction).
- **`post_backward_release_module(module)`** — called after the last backward
  node; finalizes the module and any nested FSDP units (gradient reduction +
  reshard).

The `FullyShardedDataParallelV2` adapter binds both through a single
`release_module(module, *, reduce_grad)` helper (in
`_setup_1f1b_overlap_interface`) that validates the argument is an
`FsdpModule`, then calls `module._reshard_parameter_groups()` (forward) or the
adapter's descendant-first `finalize_backward(module)` helper (backward). No
schedule-specific release helpers live on `FsdpModule` itself.

### 1.5 Root backward finalization — `post_backward()`

**File:** `combined_1f1b.py` → `FullyShardedDataParallelV2.post_backward()`

Called once after each overlapped run completes. The adapter's zero-argument
callback sweeps the root subtree with the same descendant-first helper used by
per-layer release, finalizing any `FsdpModule` that is still in the BACKWARD
phase because the schedule skipped its normal release.

`FsdpModule.post_backward()` itself remains idempotent: it finalizes only the
current FSDP unit when that unit is in the BACKWARD phase. Consequently, the
final adapter sweep safely revisits units already handled by per-layer release.
Reduce-scatter remains asynchronous; the training loop's subsequent
`finish_grad_sync()` call fences the current stream against the context's
reduce-scatter stream before any gradient consumer runs.

### 1.6 Gradient sync suppression — `no_sync_func()`

**File:** `combined_1f1b.py` → `FullyShardedDataParallelV2.no_sync()`

A context manager that suppresses gradient finalization for non-final
microbatches so gradients accumulate across inner microbatches:

```python
@contextmanager
def no_sync(self):
    self.module.context.ensure_finalized()
    context = self.module.context
    previous_state = context.is_last_microbatch
    context.is_last_microbatch = False
    try:
        yield
    finally:
        context.is_last_microbatch = previous_state
```

This wires into `config.no_sync_func` via the existing training-loop contract.

---

## 2. Fine-Grained Hook Registration

With `overlap_moe_expert_parallel_comm=True`, hooks are registered on **every
sub-module** of each FSDP unit, not just on the FSDP unit itself. These hooks
belong to `mcore_fsdp_adapter.py` because they implement the MCore combined
1F1B schedule contract, rather than the generic experimental FSDP lifecycle.

### 2.1 Pre-forward hooks on sub-modules

When the schedule calls `f_layer.attn.forward()`, a pre-forward hook on the
sub-module fires, resolves the parent `FsdpModule`, and unshards it:

```python
def _register_fine_grained_hooks(module: FsdpModule) -> None:
    for submodule in module.modules():
        if _find_fsdp_target(submodule) is not module:
            continue
        submodule.register_forward_pre_hook(_fine_grained_pre_forward,
                                            prepend=True, with_kwargs=True)

def _fine_grained_pre_forward(hook_module, args, kwargs):
    target = _find_fsdp_target(hook_module)
    if target is None:
        return
    target._unshard_parameter_groups()
    if target._unshard_event is not None:
        target.context.current_stream().wait_event(target._unshard_event)
```

### 2.2 Pre-backward hooks on sub-modules

A `register_full_backward_pre_hook` on each sub-module enters the parent
`FsdpModule` backward lifecycle before that sub-module's own backward runs, so
its weight-gradient computation sees full parameters and its later release has
a matching lifecycle/NVTX entry:

```python
def _fine_grained_pre_backward_hook(submodule: nn.Module, _grad_output) -> None:
    target = _find_fsdp_target(submodule)
    if target is None:
        return
    if target.phase is FsdpModule.Phase.RESTING:
        target.pre_backward(register_final_callback=False)
```

### 2.3 Wiring from the MCore adapter

```python
fully_shard(
    module,
    ...,
    skip_forward_backward_hooks=overlap_enabled,
)
if overlap_enabled:
    _register_fine_grained_hooks(module)
```

The adapter applies this sequence child-first to every unit it shards
(transformer layers, MoE sub-modules on the expert-DP mesh, and the root).
`skip_forward_backward_hooks=True` prevents the generic FSDP unit hooks and
parameter-completion callbacks from driving the same lifecycle as the
adapter's fine-grained hooks. State-dict protection remains registered.

---

## 3. `delay_wgrad_compute` — Explicit Schedule-Owned Completion

### Problem

The normal backward path uses per-param `post_accumulate_grad_hook`
(registered in `FsdpModule._register_hooks()`) to detect when all parameter
gradients are ready, then calls `post_backward()` which reshards and reduces.
With `delay_wgrad_compute=True`, this hook fires during `backward()`
(activation gradients only), but weight gradients are written later in
`backward_dw()`, so reduction would see partial gradients.

### Solution

The overlap adapter passes `skip_forward_backward_hooks=True` to
`fully_shard()`. `_register_hooks()` retains state-dict protection but skips
the generic forward, backward, and per-parameter completion hooks. Reduction
therefore relies entirely on the schedule's explicit
`post_backward_release_module()` call, which fires after `backward_dw()`:

```python
# In FsdpModule._register_hooks()
if skip_forward_backward_hooks:
    return
```

One integration switch is sufficient because the fine-grained schedule always
owns the complete FSDP lifecycle, whether or not delayed wgrad is enabled. This
also prevents a parameter-completion callback from racing the explicit release.

---

## 4. Public API Surface

### Modified public methods on `FsdpModule`

| Method | Signature | Description |
|---|---|---|
| `pre_backward()` | `(register_final_callback: bool = True) -> None` | Backward-phase setup; the explicit schedule disables the autograd final callback. |
| `post_backward()` | `() -> None` | Idempotently finalize the current FSDP unit: reduce, reshard, reset gradient readiness, and return to RESTING. |

### New parameters on `fully_shard()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `skip_forward_backward_hooks` | `bool` | `False` | Let an integration replace the standard forward, backward, and parameter-completion lifecycle hooks. |

### New methods on `FullyShardedDataParallelV2` (adapter)

The schedule-facing surface is assembled in `_setup_1f1b_overlap_interface`,
which binds closures that operate on a passed `FsdpModule`:

| Attribute | Signature | Description |
|---|---|---|
| `_replace_param_with_raw_if_needed` | `() -> None` | Finalize the root context; no parameter swap is needed. |
| `post_forward_release_module` | `(module) -> None` | Release forward-pass params (reshard only). |
| `post_backward_release_module` | `(module) -> None` | Descendant-first release of backward-pass params and gradients. |
| `post_backward` | `() -> None` | Final descendant-first fallback sweep after one combined segment. |
| `finish_grad_sync` | `() -> None` | Fence gradient consumers against asynchronous reduce-scatter completion. |
| `no_sync()` | `() -> contextmanager` | Suppress gradient finalization for non-final microbatches. |
| `_setup_1f1b_overlap_interface()` | `() -> None` | Bind the schedule-facing callbacks above. |

### Changes to `find_megatron_fsdp()` (`megatron_fsdp/utils.py`)

Now also detects `FullyShardedDataParallelV2` adapters (via an `FsdpModule`
wrapped by an object with `ddp_config`) alongside v1 `MegatronFSDP`.

---

## 5. Activation Recomputation Guard

Activation recomputation runs forward hooks inside backward. If the forward
hook prefetched the next module in forward order, that module's backward may
already be complete, so no later backward hook would reshard it. MFSDP v2
detects recomputation via the module phase and the active autograd GraphTask:

```python
is_recomputing = self.phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
...
if not is_recomputing:
    next_module = context.forward_order.next_item(self)
    if next_module is not None:
        next_module._unshard_parameter_groups()
```

Recomputed forwards still unshard the current module (its parameters are
needed for the recomputed GEMMs) but never prefetch a successor.

---

## 6. Files Touched

| File | Change |
|---|---|
| `experimental/module.py` | Public lifecycle APIs, optional standard lifecycle hooks, recompute guard, phase transitions |
| `experimental/fully_shard.py` | `skip_forward_backward_hooks` integration control |
| `mcore_fsdp_adapter.py` | Fine-grained combined-1F1B hooks, adapter wiring, `no_sync()`, and `_setup_1f1b_overlap_interface()` |
| `megatron_fsdp/utils.py` | Extend `find_megatron_fsdp()` |
| `tests/unit_tests/distributed/mfsdp_v2/test_context.py` | Hook opt-out and gradient-readiness reset lifecycle tests |
| `tests/unit_tests/distributed/mfsdp_v2/test_mcore_adapter.py` | Adapter-owned fine-grained hook execution and hierarchy test |
| `tests/unit_tests/distributed/mfsdp_v2/test_mcore_nd_parallel.py` | End-to-end EP-overlap parity test |
| `tests/unit_tests/distributed/mfsdp_v1/utils.py` | Shared GPT overlap-test construction and schedule-plan forward step |

---

## 7. Edge Cases

### 7.1 Activation recomputation with overlap

Handled by the phase/GraphTask detection in §5: recomputed forwards never
prefetch a successor and never reshard the current module mid-recompute.

### 7.2 Pipeline and virtual-pipeline parallelism

PP/VPP support is outside this integration. The adapter continues to reject
pipeline parallelism, and the interleaved combined schedule continues to
reject FSDP model chunks. Cross-chunk context and optimizer support belong in
follow-up changes.

---

## 8. Reference — v1 Contract

For context, the v1 API contract that the schedule calls:

| Call Site (combined_1f1b.py) | v1 Attribute/Method | Type |
|---|---|---|
| Discovery | `find_megatron_fsdp(model)` | Discovers `MegatronFSDP` instance |
| Pre-schedule | `fsdp_wrapper._replace_param_with_raw_if_needed()` | Method (root-context init for v2) |
| Microbatch loop | `no_sync_func()` → `model.no_sync()` | Context manager on adapter |
| Pre-run | `fsdp_wrapper.pre_backward()` | `partial(_root_pre_backward, module=None, skip_backward_hook=True)` |
| Layer plan | `forward_fsdp_wrapper.post_forward_release_module` | `partial(_post_forward, input=None, output=None)` |
| Layer plan | `forward_fsdp_wrapper.post_backward_release_module` | `_post_backward_release_module` |
| Post-run | `fsdp_wrapper.post_backward()` | `_root_post_backward` |
