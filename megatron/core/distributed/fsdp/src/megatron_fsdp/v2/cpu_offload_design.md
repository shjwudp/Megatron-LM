# CPU Offload Design for Megatron FSDP v2

## 1. Storage Architecture Recap

```
main_weight_buffer.data  ─── dist_param._local_tensor  (VIEW, shared Storage)
main_grad_buffer.data    ─── dist_grad._local_tensor   (VIEW, shared Storage)
model_weight_buffer.data ─── dist_param._local_tensor  (VIEW, when no main buffer)
```

- `DataParallelBuffer.data` is the backing tensor. `dist_params._local_tensor`
  and `dist_grads._local_tensor` are **sliced views** sharing the same `Storage`.
- `PyTorch` `Tensor.to("cpu")` **creates new Storage** — views do NOT follow.
  Any device move must be followed by rebuilding the views via
  `ParameterGroup._init_dist_params()`.
- `Fetch_buffer(as_shard=False)` or `self.data[offset:offset+size]`
  return the shard slice for all-gather / reduce-scatter input.

---

## 2. Goal

A single `offload_to_cpu()` call that releases **all** FSDP-held GPU memory:

| Resource | Offloaded? | Notes |
|----------|-----------|-------|
| `model_weight_buffer.data` | Yes | Shard used for all-gather every micro-batch |
| `transpose_weight_buffer.data` | Yes | MXFP8 columnwise data |
| `main_weight_buffer.data` | Yes | fp32 optimizer weights |
| `main_grad_buffer.data` | Yes | Gradient accumulation buffer |
| `TracePoolAllocator` slot tensors | Yes | Via `allocator.release()` |
| Temp `_unsharded_buffer` | No | Already freed by `reshard()` after every micro-batch |

All buffers **auto-reload** to GPU when accessed by the next FSDP operation
(`unshard`, `reduce_grad`, `_copy_main_weights_to_model_weights`, etc.).

---

## 3. The View Problem

`self.data = self.data.to("cpu")` creates a new tensor with new `Storage`.
Any existing `dist_params._local_tensor` views still point at the **old**
GPU Storage — they become dangling.

**Constraint**: The optimizer holds references to the original `dist_param`
DTensor objects in its `param_groups`.  We CANNOT replace `dist_params` with
new DTensors — the optimizer would lose its connection.  Instead, we must
update the **`_local_tensor` inside** each existing DTensor in-place.

**Solution**: After every device move, re-slice fresh views from the buffer
and swap them into `dist_param._local_tensor` (and `dist_grad._local_tensor`)
directly, **without** creating new DTensor objects.

```python
# DataParallelBuffer — low-level device move, optional pinned memory for speed
def _move_data_to(self, target_device: torch.device, pin_memory: bool = False) -> None:
    """Move self.data to target_device.  Caller must rebuild dependent views."""
    if self.data is None or self.data.device == target_device:
        return
    if target_device.type == "cpu" and pin_memory:
        # Allocate pinned CPU memory and copy — much faster GPU→CPU via DMA
        cpu_data = torch.empty(self.data.shape, dtype=self.data.dtype,
                               pin_memory=True)
        cpu_data.copy_(self.data, non_blocking=True)
        self.data = cpu_data
    else:
        self.data = self.data.to(target_device, non_blocking=True)
```

```python
# ParameterGroup — update _local_tensor in-place on existing DTensors
def _rebuild_dist_views(self) -> None:
    """Re-slice dist_params/dist_grads ._local_tensor to point at current buffer data.

    Does NOT create new DTensor objects — updates _local_tensor in-place
    so existing optimizer param_group references remain valid.
    """
    is_param_shard = self.sharding_strategy in ("optim", "optim_grads", "optim_grads_params")

    for i, param in enumerate(self.params):
        dist_param = self.dist_params[i]
        if dist_param is not None:
            # Re-slice view from current buffer data
            if self.main_weight_buffer is not None:
                data = self.main_weight_buffer.get_item(self.param_idx[param], as_shard=is_param_shard)
            elif self.model_weight_buffer is not None:
                data = self.model_weight_buffer.get_item(self.param_idx[param], as_shard=is_param_shard)
            else:
                continue
            # In-place update — optimizer references remain valid
            object.__setattr__(dist_param, '_local_tensor', data)

    # Rebuild dist_grads views similarly
    if self.main_grad_buffer is not None:
        is_grad_shard = is_param_shard
        for i, param in enumerate(self.params):
            dist_grad = self.dist_grads[i]
            if dist_grad is not None:
                grad_data = self.main_grad_buffer.get_item(self.param_idx[param], as_shard=is_grad_shard)
                object.__setattr__(dist_grad, '_local_tensor', grad_data)
```

---

## 4. API

```python
class FSDPModule:
    def offload_to_cpu(self, recursive: bool = True, pin_memory: bool = False) -> None:
        """Offload all FSDP-held GPU memory to CPU.

        Moves every DataParallelBuffer.data to CPU and releases
        TracePoolAllocator slot tensors.  All buffers auto-reload
        on next access — no explicit reload call needed.

        Args:
            recursive: If True (default), also offloads child FSDPModules.
            pin_memory: If True, allocate pinned CPU memory for faster
                CPU↔GPU transfers (~12 GB/s via DMA vs ~3-6 GB/s pageable).
                Costs CPU RAM that the OS cannot swap out.
        """

    def reload_to_gpu(self, recursive: bool = True) -> None:
        """Explicitly prefetch all buffers back to GPU.

        Normally not needed — every access path auto-reloads.
        Useful to hide the first-touch CPU→GPU copy latency.
        """
```

---

## 5. `offload_to_cpu()` Implementation

```python
def offload_to_cpu(self, recursive: bool = True, pin_memory: bool = False) -> None:
    modules = self._get_fsdp_modules(recursive)
    ctx = self._fsdp_root_context

    # 1. Offload all DataParallelBuffer data tensors to CPU
    for module in modules:
        for pg in module._fsdp_param_groups:
            for buf in (pg.model_weight_buffer, pg.transpose_weight_buffer,
                        pg.main_weight_buffer, pg.main_grad_buffer):
                if buf is not None and buf.data is not None:
                    buf._move_data_to(torch.device("cpu"), pin_memory=pin_memory)
            pg._rebuild_dist_views()  # in-place _local_tensor update

    # 2. Release allocator slot tensors (auto-resume on next alloc/free)
    if isinstance(ctx.bucket_allocator, TracePoolAllocator):
        ctx.bucket_allocator.release()
```

---

## 6. Auto-Reload Mechanism

`DataParallelBuffer` methods that need GPU data call `_ensure_data_on_gpu()`.
After the reload, `ParameterGroup._rebuild_dist_views()` updates
`dist_param._local_tensor` / `dist_grad._local_tensor` in-place so they
point at the new GPU Storage — **without** creating new DTensor objects
(the optimizer's references stay valid).

```python
class DataParallelBuffer:
    # Target GPU device (stored at init time)
    _gpu_device: torch.device

    def _is_on_cpu(self) -> bool:
        return self.data is not None and self.data.device.type == "cpu"

    def _ensure_data_on_gpu(self) -> bool:
        """Move data to GPU if offloaded. Returns True if a move happened.
        When source is pinned CPU memory, non_blocking=True uses DMA
        and can overlap with other CUDA streams."""
        if not self._is_on_cpu():
            return False
        self.data = self.data.to(self._gpu_device, non_blocking=True)
        return True

    def unshard(self, bind_params: bool = False) -> torch.Tensor:
        if self._ensure_data_on_gpu():
            self._param_group._rebuild_dist_views()
        # ... existing unshard logic ...
```

### Auto-reload points

| Entry point | Buffer | Method |
|-------------|--------|--------|
| `DataParallelBuffer.unshard()` | `model_weight`, `transpose_weight` | All-gather needs GPU shard |
| `DataParallelBuffer.reduce_grad()` | `main_grad` | Reduce-scatter needs GPU tensor |
| `DataParallelBuffer.fetch_buffer()` | any | Caller expects GPU tensor |
| `_copy_main_weights_to_model_weights` path | `main_weight` | Quantization needs GPU tensor |

All of these call `_ensure_data_on_gpu()` + `_rebuild_dist_views()` if a
reload occurred.

---

## 7. Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│ TRAINING STEP                                                        │
│                                                                      │
│   offload_to_cpu()         ← user calls (all buffers → CPU)          │
│   ┌───────────────────┐                                              │
│   │ Other app uses GPU │  (checkpoint, inference, etc.)               │
│   └───────────────────┘                                              │
│                                                                      │
│   unshard()                                                          │
│     └─ model_weight_buffer._ensure_data_on_gpu()  ← auto-reload      │
│     └─ _rebuild_dist_views()                                         │
│     └─ all_gather(shard) → full GPU buffer                           │
│                                                                      │
│   forward / backward     (GPU compute, full buffer)                   │
│                                                                      │
│   reshard()              (frees _unsharded_buffer, model_weight_buffer│
│                           .data now on GPU — not re-offloaded)        │
│                                                                      │
│   reduce_grad()                                                      │
│     └─ main_grad_buffer._ensure_data_on_gpu()     ← auto-reload      │
│     └─ _rebuild_dist_views()                                         │
│     └─ copy .grad → main_grad; reduce-scatter                        │
│                                                                      │
│   optimizer.step()       (reads main_weight + main_grad,             │
│                           dist_param references still valid —         │
│                           rebuild only updated _local_tensor)         │
│                                                                      │
│   _copy_main_weights_to_model_weights()                              │
│     └─ main_weight_buffer._ensure_data_on_gpu()   ← auto-reload      │
│     └─ _rebuild_dist_views()                                         │
│     └─ quantize main_weight → model_weight                           │
│                                                                      │
│   offload_to_cpu()         ← user calls (all buffers → CPU)          │
│                                                                      │
│   (next step...)                                                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Cost per step**: 3 CPU→GPU copies (model_weight shard, main_grad shard,
main_weight shard). Each copy is `1/dp_size` of full model — for dp=8,
this is ~12.5% of model size, or ~9 GB for a 70B BF16 model.

With `pin_memory=True` (DMA ~12 GB/s): ~0.75 s per 9 GB copy.
With pageable memory (~3-6 GB/s): ~1.5-3 s per copy.
Pinned memory adds CPU-side memory pressure (page-locked, not swappable).

`TracePoolAllocator` slots auto-resume on the first `allocate()` / `free()`
call (existing `_auto_resume()`).

---

## 8. Memory Savings

| Component | Size (per DP rank) | Saved by offload_to_cpu() |
|-----------|-------------------|---------------------------|
| `model_weight_buffer.data` | `numel * elem_size / dp` | Same (shard moves to CPU) |
| `main_weight_buffer.data` | `numel * 4 / dp` (fp32) | Same |
| `main_grad_buffer.data` | `numel * 4 / dp` (fp32) | Same |
| `transpose_weight_buffer.data` | MXFP8-only, same as model | Same |
| `TracePoolAllocator` slots | ~2× model_weight (full all-gather + grad buffers) | Total freed via `release()` |

**Example** (70B BF16 model, dp=8):
- `model_weight_buffer`: 70B × 2 / 8 = 17.5 GB → freed
- `main_weight_buffer`: 70B × 4 / 8 = 35 GB → freed
- `main_grad_buffer`: 70B × 4 / 8 = 35 GB → freed
- Allocator slots: ~40 GB → freed via `release()`
- **Total freed**: ~127 GB GPU → available for other apps

---

## 9. `_rebuild_dist_views()` Detail

After `self.data` is moved, `dist_param._local_tensor` views are stale.
`_rebuild_dist_views()` re-slices fresh views from the buffer and swaps
them in-place using `object.__setattr__`.

### Why in-place?

The optimizer's `param_groups` hold references to the original `dist_param`
DTensor objects.  Creating new DTensors would break these references,
causing the optimizer to optimize dead tensors.  In-place `_local_tensor`
update preserves the DTensor identity.

### What about `_replace_module_parameter`?

Not needed during offload/reload.  Between micro-batches (where
`offload_to_cpu` / `reload_to_gpu` are called), the module already holds
`dist_params` (the sharded state).  `_replace_module_parameter` is only
needed during `unshard()` (swap in full params) and `reshard()` (swap in
sharded dist_params) — neither is affected by offload/reload.

---

## 10. `reload_to_gpu()` — Explicit Prefetch

```python
def reload_to_gpu(self, recursive: bool = True) -> None:
    """Explicitly move all buffers back to GPU and rebuild views.

    Useful to hide CPU→GPU copy latency before a training step,
    especially with pin_memory=True where non_blocking copies can
    overlap with other work.  Otherwise auto-reload handles it
    transparently.
    """
    modules = self._get_fsdp_modules(recursive)
    for module in modules:
        for pg in module._fsdp_param_groups:
            for buf in (pg.model_weight_buffer, pg.transpose_weight_buffer,
                        pg.main_weight_buffer, pg.main_grad_buffer):
                if buf is not None and buf.data is not None:
                    buf._move_data_to(module.device)
            pg._rebuild_dist_views()
```

---

## 11. Future: `CPUOffloadPolicy` — torch FSDP2-Style

For deeper integration where the optimizer always runs on CPU:

```python
@dataclass
class CPUOffloadPolicy:
    offload_weights: bool = True
    offload_grads: bool = True
    pin_memory: bool = True

fully_shard(module, cpu_offload_policy=CPUOffloadPolicy())
```

Differences from the bulk `offload_to_cpu()`:
- Buffers allocated on **pinned CPU** from `_init_buffers()` (no initial GPU alloc)
- `_copy_main_weights_to_model_weights()` copies shard CPU→GPU, quantizes,
  writes to `model_weight_buffer`, then `model_weight_buffer` stays on GPU
- After optimizer step, `main_weight_buffer` stays on CPU (no re-offload needed)
- Optimizer **always** runs on CPU — saves optimizer state GPU memory too
- More code changes (~200 lines) touching `param_group.py`, `dp_buffer.py`,
  `mixed_precision.py`

This is a follow-up feature; the bulk `offload_to_cpu()` above is the
immediate deliverable.

---

## 12. Implementation Plan

| Step | What | Files |
|------|------|-------|
| 1 | `DataParallelBuffer._move_data_to()` + `_is_on_cpu()` + `_ensure_data_on_gpu()` | `dp_buffer.py` |
| 2 | `ParameterGroup._rebuild_dist_views()` — in-place `_local_tensor` update | `param_group.py` |
| 3 | `FSDPModule._get_fsdp_modules(recursive)` | `fsdp_module.py` |
| 4 | `FSDPModule.offload_to_cpu(recursive)` | `fsdp_module.py` |
| 5 | `FSDPModule.reload_to_gpu(recursive)` | `fsdp_module.py` |
| 6 | Auto-reload: `_ensure_data_on_gpu()` in `unshard`, `reduce_grad`, `fetch_buffer` | `dp_buffer.py` |
| 7 | Auto-reload: `_ensure_data_on_gpu()` in `_copy_main_weights_to_model_weights` path | `fsdp_module.py` / `mixed_precision.py` |
| 8 | Unit tests: offload → auto-reload → correctness | `test_allocator.py` + new |

---

## 13. Risks

| Risk | Mitigation |
|------|-----------|
| View invalidation after device move | Always call `_rebuild_dist_views()` after `_move_data_to()` |
| Optimizer loses param references | `_rebuild_dist_views()` uses `object.__setattr__` on existing DTensors — optimizer references stay valid |
| Stale `_local_tensor` after auto-reload | Each method that calls `_ensure_data_on_gpu()` also calls `_rebuild_dist_views()` |
| Performance regression from view rebuilds | `_rebuild_dist_views()` is O(num_params × num_groups), ~0.1ms per group; 3 rebuilds per step is negligible |
