# Lazy grad_buffer management in Megatron FSDP v2

`grad_buffer` is the `DataParallelBuffer` owned by each `ParameterGroup`
for optimizer-facing gradients.

During backward, Megatron FSDP v2 stages local parameter gradients into this
buffer, runs the required data-parallel collective, and exposes DTensor views of
the reduced result through `optimizer_param.grad` or `optimizer_param.decoupled_grad`.
The optimizer consumes those DTensor gradient views.

Lazy management means the buffer layout is created during FSDP initialization,
but the backing tensor is allocated only when gradients are first produced. When
`zero_grad(set_to_none=True)` clears optimizer-facing gradient references, the
backing tensor can be released. When `zero_grad(set_to_none=False)` is used,
the existing backing tensor is zeroed in place and kept allocated.

## Core idea

`ParameterGroup._init_buffers()` creates the `DataParallelBuffer` metadata for
gradients when the parameter group requires gradients:

```python
if self.requires_grad:
    main_grads_dtype = self.mp_policy.main_grads_dtype_for_param(self.params[0])
    self.grad_buffer = self._create_buffer(main_grads_dtype, "main_grad")
```

At this point the buffer has layout metadata (`BufferIndex`, shard sizes,
parameter offsets), but `grad_buffer.data` is still `None`. The
corresponding `optimizer_grads` entries are placeholders.

`ParameterGroup.prepare_gradient_storage()` performs the deferred allocation:

1. return immediately if the group has no grad buffer, does not require grads,
   or the buffer is already allocated;
2. allocate `grad_buffer.data` with `torch.empty(...)`;
3. slice the buffer according to the active sharding layout;
4. build DTensor gradient views in `optimizer_grads` on first use, or rebind cached
   wrappers to the new local slices after a prior storage release.

The DTensor views are what the optimizer later sees through
`optimizer_param.grad` or `optimizer_param.decoupled_grad`.

Reusing wrappers avoids rebuilding the complete Python DTensor object graph on
every iteration. `ParameterGroup._optimizer_grads` owns those wrappers while
storage is absent. During that detached interval, public `optimizer_grads` entries
remain `None`; a DTensor whose local tensor has been detached is never exposed
to optimizer or checkpoint code. The uneven-DTensor module centralizes the
private `_local_tensor` detach/rebind operation, including local-shape recovery
and checkpoint chunk-metadata restoration. This is the only lifecycle code
in the lazy-gradient detach/rebind path that should depend on PyTorch's private
DTensor field.

## Normal lifecycle

| Point in step | Behavior |
| --- | --- |
| FSDP initialization | Create `grad_buffer` metadata only. `grad_buffer.data` is `None`; `optimizer_grads` contains placeholders. |
| First backward staging | `prepare_gradient_storage()` allocates `grad_buffer.data`, establishes the direct DDP/ZeRO-1 full-gradient view, and rebuilds `optimizer_grads` DTensor views. |
| Gradient reduction | DDP and ZeRO-1 stage directly into persistent full-gradient storage; sharded-gradient strategies use full scratch. All-reduce or reduce-scatter writes the optimizer-facing result. |
| Optimizer step | Optimizer consumes `optimizer_param.grad` or `optimizer_param.decoupled_grad`, which are backed by `grad_buffer.data`. |
| `zero_grad(set_to_none=True)` | Clear optimizer-facing gradient references, unbind any direct DDP/ZeRO-1 full-gradient view, privately cache and detach reusable DTensor wrappers, reset accumulation state, and release `grad_buffer.data` if nothing still references valid gradients. |
| `zero_grad(set_to_none=False)` | Keep `grad_buffer.data` and any direct full-gradient view allocated and zero storage in place. |

`release_grad_storage_if_unused()` is also called from the forward pre-hook.
That call is idempotent and handles the common case where `zero_grad()` has
already cleared all optimizer-facing gradient references before the next
forward.

The normal root forward additionally calls
`FSDPModule._release_grad_storage_if_unused()` before the first parameter
unshard. The module method only traverses the root's parameter groups and
delegates to `ParameterGroup.release_grad_storage_if_unused()`. This root-wide
sweep supports plain PyTorch optimizers: their
`zero_grad(set_to_none=True)` clears `optimizer_param.grad` but does not call the
FSDP module's zero-grad method, leaving parameter-group accumulation flags from
the previous step. Each parameter group independently retains live or
accumulating gradients and resets stale state through `ParameterGroup.zero_grad()`
when its optimizer-facing gradients have been cleared.

The root-wide sweep is outside the full-iteration CUDA graph lifecycle. When
`enable_full_iteration_cuda_graph=True`, each parameter-group guard returns
before gradient-liveness inspection and never calls `ParameterGroup.zero_grad()`.
Full-iteration mode keeps graph-visible gradient objects and owns its in-place
zeroing separately.

Doing this at the root boundary is important: the older per-module release
path ran after each module unshard, so later-layer gradient shards could overlap
the next forward's parameter all-gathers and activations. The per-module call
remains as an idempotent fallback for schedules that invoke child FSDP modules
directly.

## Release guard

`release_grad_storage_if_unused()` frees `grad_buffer.data` only when all
of these are true:

- full-iteration CUDA graph mode is not enabled for the group;
- `grad_buffer.data` exists;
- the gradient phase is not `ACCUMULATING`;
- no `optimizer_param.grad` or `optimizer_param.decoupled_grad` still references the
  gradient DTensor.

If any of those conditions fail, storage is kept because it may still be needed
by gradient accumulation or the optimizer.

On release, each live `optimizer_grads` wrapper moves to the private cache before
its local tensor reference is removed, and any direct full-gradient view is
cleared before `grad_buffer` is unbound. `optimizer_grads` is then reset to
`None` placeholders. On the next `prepare_gradient_storage()`, the backing
buffer is allocated, the cached wrappers are rebound to correctly shaped local
slices, uneven chunk metadata is restored from the corresponding distributed
parameter, and only then are the wrappers published through `optimizer_grads`
again.

The wrapper layout is immutable for the lifetime of a parameter group. Rebind
validates shape, dtype, device, mesh, placements, and checkpoint metadata the
first time a cached wrapper is reused. Later iterations update the fixed-size
cache and live-gradient lists in place and skip repeated structural validation;
the backing buffer may change, but its established layout contract does not.

## Accumulation state

`GradientPhase` records whether the persistent gradient is `EMPTY`,
`ACCUMULATING`, or optimizer-`READY`. Placement determines where accumulation
happens. When `grad_storage` is fully replicated and `grad_accumulation` is
fully partial, DDP and ZeRO-1 accumulate local microbatches directly in
`grad_buffer`; `full_grad_has_value` is then derived from the
`ACCUMULATING` phase. ZeRO-2, FSDP, and HSDP reduce every microbatch into their
configured accumulation placement instead.

`zero_grad()` resets the phase before trying to release storage. The phase is
required because placement alone cannot distinguish a partial accumulation
from an optimizer-ready value, and some ranks can have empty local optimizer
shards while the shared buffer still contains valid data.

## Safe use of `torch.empty`

The lazy allocation uses `torch.empty()` to avoid an unnecessary zero-fill.
This is safe because the first use after allocation is controlled by the
accumulation flags:

- if no previous gradient has accumulated, staging and collective outputs
  overwrite the destination;
- if a previous microbatch has accumulated, DDP and ZeRO-1 stage by addition
  into persistent full storage, while reduced-gradient strategies add their
  collective output into the configured accumulation placement.

`zero_grad(set_to_none=False)` is the explicit keep-storage path: it zeros
`grad_buffer.data` in place when the buffer exists instead of releasing it.

## CUDA graph exceptions

Full-iteration CUDA graph mode keeps optimizer-facing gradient storage alive so
the captured step can reuse stable gradient objects. In that mode,
both the root-wide and per-module
`release_grad_storage_if_unused()` calls return without scanning or freeing
the buffer, and the full-iteration optimizer lifecycle clears the existing
storage in place.

Per-module CUDA graph capture keeps the normal lazy behavior, except compatible
main-grad storage may be initialized before capture so trace and replay use the
same buffer surface.

## Relevant code

| File | Relevant pieces |
| --- | --- |
| `param_group.py` | `_init_buffers()`, `prepare_gradient_storage()`, `release_grad_storage_if_unused()`, `zero_grad()` |
| `fsdp_module.py` | `reduce_grad()` installs optimizer-facing gradients; `_release_grad_storage_if_unused()` traverses the root's parameter groups |
| `hooks.py` | Root-before-unshard and per-module gradient-storage release paths, plus CUDA-graph pre-initialization |
