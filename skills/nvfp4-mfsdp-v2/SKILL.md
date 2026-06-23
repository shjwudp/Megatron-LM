---
name: nvfp4-mfsdp-v2
description: NVFP4 development in Megatron FSDP v2. Covers shape handling (logical vs storage), BufferIndex.compact, chunk_size_factor, get_param_storage_shapes, checkpoint testing, and common pitfalls. Use when adding NVFP4 features, debugging NVFP4 shape mismatches, writing NVFP4 tests, or modifying buffer layout logic.
user_invocable: true
argument: "<topic>  # shape-handling | compact | testing | all"
---

# NVFP4 Development in MFSDP v2

NVFP4 stores 2 values per byte in uint8, so the **storage shape** differs from
the **logical shape** (e.g., logical `[128, 128]` → packed storage `[128, 64]`).
This distinction cascades through buffer layout, DTensor creation, and
checkpoint logic.

---

## Shape Handling Architecture

### Key Rule

Only the `model_weight_buffer` and `transpose_weight_buffer` use packed
(storage) shapes. The `main_weight_buffer` and `main_grad_buffer` hold fp32
data and **always** use logical shapes.

### `get_param_storage_shapes(params)`

Defined on `MixedPrecisionPolicy` in `mixed_precision.py`.
Returns the per-param shapes for NVFP4 model-weight storage:

```python
def get_param_storage_shapes(self, params):
    """Return packed shapes for NVFP4, logical shapes otherwise."""
    if not HAVE_TE_NVFP4 or not any(self.is_nvfp4_param(p) for p in params):
        return [p.shape for p in params]
    shapes = []
    for p in params:
        if self.is_nvfp4_param(p):
            packed = list(p.shape)
            packed[-1] = packed[-1] // 2
            shapes.append(torch.Size(packed))
        else:
            shapes.append(p.shape)
    return shapes
```

Call sites and their shape expectations:

| Call site | Buffer type | Shape expected |
|-----------|------------|----------------|
| `DataParallelBuffer.__init__` → compact | model_weight/transpose_weight | packed (NVFP4) |
| `_init_dist_params` (model_weight branch) | DTensor for model weight | packed |
| `_init_dist_params` (main_weight branch) | DTensor for main weight | logical (`param.shape`) |
| `_init_dist_params` (grad loop) | DTensor for gradients | logical (`p.shape`) |

---

## BufferIndex.compact Pattern

### Motivation

All buffers share the same `chunk_size_factor` (computed from logical shapes)
so that item offsets scale proportionally across buffers. The model-weight
buffer then calls `compact(0.5, ...)` to shrink its metadata without changing
the proportional mapping.

### Flow in DataParallelBuffer.__init__

```python
# Step 1: build BufferIndex with logical shapes + shared chunk_size_factor
_logical_shapes = [p.shape for p in params]
self.buffer_index = BufferIndex(
    param_shapes=_logical_shapes,
    chunk_size_factor=chunk_size_factor,
    ...,
)

# Step 2: compact NVFP4 weight buffers
if buffer_role in ("model_weight", "transpose_weight") and any(
    mp_policy.is_nvfp4_param(p) for p in params
):
    compact_shapes = mp_policy.get_param_storage_shapes(params)
    self.buffer_index.compact(0.5, compact_shapes)
```

### BufferIndex.compact Implementation

```python
def compact(self, factor: float, compact_shapes: List[torch.Size]) -> None:
    new_map = {}
    for item_id, item in self.item_index_map.items():
        new_map[item_id] = ItemIndex(
            global_data_index=int(item.global_data_index * factor),
            size=int(item.size * factor),
            item_id=item.item_id,
            shape=compact_shapes[item_id],
        )
    self.item_index_map = new_map
    self.bucket_meta = BucketMeta(
        global_data_index=0,
        size=int(self.bucket_meta.size * factor),
        items=list(new_map.values()),
    )
    self.shard_meta = self._build_shard_meta(
        self.bucket_meta, self.is_distributed, self.dp_world_size, self.dp_rank
    )
```

BufferIndex stores `dp_rank`, `dp_world_size`, `chunk_size_factor`,
`sharding_strategy` as instance attributes (needed for shard rebuild).

### Shape Access (no _item_shapes)

After removal of `_item_shapes` from `DataParallelBuffer`, shapes are accessed
directly from the BufferIndex:

```python
# In main_grad_getter (fsdp_module.py):
param_shape = gbuf.buffer_index.item_index_map[item_id].shape
grad_data = gbuf_data[offset : offset + size].view(param_shape)
```

---

## chunk_size_factor

Computed in `ParameterGroup.__init__` from **logical** shapes:

```python
if len(params) > 0 and any(p.shape[1:].numel() > 0 for p in params):
    self.chunk_size_factor = max(1, math.lcm(*[p.shape[1:].numel() for p in params]))
```

This single value is passed to all three buffers. Using packed shapes for the
LCM would misalign the main (fp32) buffers.

| Buffer | chunk_size_factor | Shapes in BufferIndex |
|--------|-------------------|----------------------|
| model_weight | logical (shared) | compacted to packed (after compact) |
| main_weight | logical (shared) | logical |
| main_grad | logical (shared) | logical |

---

## NVFP4 Checkpoint Testing

### Test Structure

Round-trip tests (v2 → v2) save a checkpoint from a trained NVFP4 model and
load it back, verifying model weights and optimizer state match exactly.

### Required Config

NVFP4 requires `bf16=True` (RHT quantization only supports bfloat16).

```python
dict(
    data_parallel_sharding_strategy="optim_grads_params",
    fp4="e2m1",
    fp4_recipe="nvfp4",
    fp4_param_gather=True,
    bf16=True,
)
```

### NVFP4 Availability Check

```python
try:
    from transformer_engine.pytorch.fp8 import check_nvfp4_support
    _NVFP4_AVAILABLE, _NVFP4_SKIP_REASON = check_nvfp4_support()
except Exception:
    _NVFP4_AVAILABLE = False
    _NVFP4_SKIP_REASON = "NVFP4 support unavailable"
```

Apply `pytest.mark.skipif` per-parametrize:

```python
pytest.param(
    ...,
    marks=pytest.mark.skipif(not _NVFP4_AVAILABLE, reason=_NVFP4_SKIP_REASON),
    id="v2_rt_nvfp4_optim_grads_params",
),
```

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Shape mismatch at `main_grad.copy_(...)` | main_grad_buffer uses packed shapes but param.grad has logical shape | Ensure only model_weight/transpose_weight buffers use packed shapes |
| `RHT is only supported for bfloat16` | NVFP4 quantization requires bf16 input | Add `bf16=True` to test/model config |
| Buffer too small for NVFP4 params | `chunk_size_factor` computed from packed shapes misaligns main buffers | Use logical `p.shape[1:]` for chunk_size_factor |
| DTensor shape mismatch | DTensor created with wrong shape for the backing buffer data | Use packed shape for model_weight data, logical shape for main_weight/grad data |
| `_item_shapes` AttributeError | Recent refactor removed `_item_shapes` from DataParallelBuffer | Use `buffer_index.item_index_map[item_id].shape` instead |

---

## Files Reference

| File | Role in NVFP4 |
|------|---------------|
| `megatron_fsdp/v2/dp_buffer.py` | BufferIndex, compact, DataParallelBuffer shape logic |
| `megatron_fsdp/v2/mixed_precision.py` | `get_param_storage_shapes`, `FullyShardNVFP4Policy`, NVFP4 detection |
| `megatron_fsdp/v2/param_group.py` | chunk_size_factor, `_init_dist_params` shape selection |
| `megatron_fsdp/v2/fsdp_module.py` | `main_grad_getter` shape access |
| `tests/unit_tests/distributed/megatron_fsdp/v2/test_mcore_checkpoint.py` | NVFP4 round-trip checkpoint test |
| `megatron_fsdp/v2/nvfp4_design.md` | Full design document |
