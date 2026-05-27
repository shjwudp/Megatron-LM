# Megatron FSDP2 Checkpoint Design

## 1. Overview

This document describes the checkpoint save/load design for Megatron FSDP2
(`use_fully_shard_api=True`). Megatron FSDP2 wraps model parameters as PyTorch
`DTensor` objects and shards them across the data-parallel dimension. This is
distinct from PyTorch's own `torch.distributed.fsdp2` (referred to as "torch
FSDP2" below). Checkpointing must handle these sharded DTensors correctly,
including support for the Megatron `DistributedOptimizer` and online checkpoint
format conversion.

### Goals

- Save and load Megatron FSDP2 model + optimizer state via PyTorch DCP
  (`torch.distributed.checkpoint`).
- Support `fsdp_dtensor` as the canonical checkpoint format for Megatron FSDP2.
- Enable online checkpoint conversion from legacy Megatron formats
  (ND-parallel, Megatron FSDP v1 baseline) to Megatron FSDP2.
- Handle uneven DTensor sharding (parameters not evenly divisible by DP size).
- Preserve compatibility with existing Megatron checkpoint infrastructure.

### Non-Goals (current scope)

- Async checkpoint save/load for Megatron FSDP2.
- Cross-node checkpoint resharding (different DP topology on load).

---

## 2. Background: Two FSDP Paths

MCore wraps FSDP through `FullyShardedDataParallel` (in `mcore_fsdp_adapter.py`).
There are two code paths:

| Path | Flag | Inner module | Model state dict |
|------|------|-------------|------------------|
| **Legacy Megatron FSDP** | `use_megatron_fsdp=True`, `use_fully_shard_api=False` | `MegatronFSDP` | `state_dict()` with DTensor hooks |
| **Megatron FSDP2** | `use_megatron_fsdp=True`, `use_fully_shard_api=True` | `FSDPModule` (DTensor-native) | `state_dict()` (DTensors natively) |

The `fsdp_dtensor` checkpoint format (`--ckpt-format fsdp_dtensor`) is the required
format for both paths. It uses DCP directly, storing each parameter as a `DTensor`.

---

## 3. Architecture

### 3.1 Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Megatron Training Loop                    │
│  checkpointing.py: save_checkpoint / load_checkpoint        │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────────┐
│ Model State │ │ Optim State │ │ Scheduler / RNG │
│   Dict      │ │   Dict      │ │   State         │
└──────┬──────┘ └──────┬──────┘ └─────────────────┘
       │               │
       ▼               ▼
┌─────────────────────────────────────────────────────────────┐
│         preprocess_fsdp_dtensor_state_dict()                │
│  1. handle_fp8_extra_state_case                             │
│  2. handle_swiglu_in_state_dict (model + optimizer)         │
│  3. handle_experts_in_state_dict (EP key remapping)         │
│  4. preprocess_state_dict_for_uneven_dtensor                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            torch.distributed.checkpoint.save/load           │
│         (DCP handles DTensor serialization natively)        │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Module Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| `uneven_dtensor.py` | `megatron/core/distributed/fsdp/src/megatron_fsdp/` | `get_state_dict`, `preprocess_state_dict_for_uneven_dtensor`, chunk metadata for uneven DTensors |
| `fsdp_dtensor_checkpoint.py` | `megatron/core/transformer/` | SWiGLU split, GDN split, expert key remapping, FP8 cleanup |
| `distrib_optimizer.py` | `megatron/core/optimizer/` | `state_dict()`, `load_state_dict()`, `sharded_state_dict()`, `sharded_param_state_fsdp_dtensor()` |
| `checkpointing.py` | `megatron/training/` | High-level save/load orchestration, `preprocess_fsdp_dtensor_state_dict()` |
| `mcore_fsdp_adapter.py` | `megatron/core/distributed/fsdp/` | `MegatronFSDPAdapter` — routes to v1 `MegatronFSDP` or Megatron FSDP2 `fully_shard` |

---

## 4. State Dict Flow

### 4.1 Model State Dict

Megatron FSDP2 uses `model.state_dict()` which returns a dict of `DTensor` values.
The keys follow the Megatron `module.` prefix convention
(e.g., `module.embedding.word_embeddings.weight`).

For the `DistributedOptimizer` + Megatron FSDP2 path, the model state dict is
obtained via `model.state_dict_for_save_checkpoint()`.

**Current status:** `state_dict_for_save_checkpoint` is set to `not_implemented_op`
for v2 (line 384 of `mcore_fsdp_adapter.py`). Fix: wire to `model.state_dict()`
(already produces DTensors).

```python
# In _init_with_fully_shard(), replace:
self.module.state_dict_for_save_checkpoint = not_implemented_op
# With:
self.module.state_dict_for_save_checkpoint = lambda *args, **kwargs: module.state_dict()
self.state_dict_for_save_checkpoint = lambda *args, **kwargs: module.state_dict()
```

**Key detail:** Megatron FSDP2 parameters are `DTensor` objects. DCP serializes
them natively — each rank writes its local shard, and DCP handles the global
metadata.

### 4.2 Optimizer State Dict

There are **two** paths for obtaining the optimizer state dict, depending on
the context:

#### Path A: Megatron Training Loop (`checkpointing.py`)

Uses `optimizer.sharded_state_dict(model_state_dict, sharding_type="fsdp_dtensor")`,
which calls `sharded_param_state_fsdp_dtensor()`. Returns:

```python
{
    "state": {
        "<param_name>": {"exp_avg": DTensor(...), "exp_avg_sq": DTensor(...)},
        ...
    },
    "param_to_group_meta": {
        "<param_name>": {"lr": ..., "weight_decay": ...},
        ...
    }
}
```

This is the **primary path** for Megatron-integrated training.

#### Path B: Standalone `get_state_dict()` (`uneven_dtensor.py`)

Uses PyTorch's native `torch.distributed.checkpoint.state_dict.get_state_dict()`,
which calls `optimizer.state_dict()` internally. For `DistributedOptimizer`
with Megatron FSDP, `state_dict()` returns the inner optimizer's full state dict
directly (see Section 5.2).

This path is used by:
- `test_checkpoint_online_convert.py` (online format conversion tests)
- `fsdp_toy.py` example (standalone FSDP training)
- Any code using the `AppState` wrapper pattern

### 4.3 Save Flow

```
save_checkpoint()
  |
  +-- 1. generate_state_dict()
  |     |
  |     +-- model[i].state_dict_for_save_checkpoint()
  |     |     Returns DTensor state dict.
  |     |
  |     +-- optimizer.sharded_state_dict(state_dict, metadata={'distrib_optim_sharding_type': 'fsdp_dtensor'})
  |     |     Returns {"state": {name: opt_state}, "param_to_group_meta": {...}}
  |     |     via sharded_param_state_fsdp_dtensor().
  |     |
  |     +-- rng_state, scheduler state, etc.
  |
  +-- 2. preprocess_fsdp_dtensor_state_dict()
  |     - handle_fp8_extra_state_case
  |     - handle_swiglu_in_state_dict
  |     - handle_experts_in_state_dict
  |     - preprocess_state_dict_for_uneven_dtensor
  |
  +-- 3. torch.distributed.checkpoint.save(state_dict, storage_writer)
```

### 4.4 Load Flow

```
load_checkpoint()
  |
  +-- 1. Build sharded_state_dict via generate_state_dict()
  |     Same structure as save. For optimizer: is_loading=True triggers
  |     _init_optimizer_states_with_dummy_values() to create placeholder states.
  |
  +-- 2. _load_base_checkpoint()
  |     - preprocess_fsdp_dtensor_state_dict()
  |     - torch.distributed.checkpoint.load_state_dict(state_dict, reader)
  |
  +-- 3. Post-load application
        - ddp_model[i].load_state_dict(state_dict['model'], strict)
        - optimizer.load_state_dict(state_dict['optimizer'])
        - RNG states, scheduler, etc.
```

---

## 5. DistributedOptimizer Integration

### 5.1 FSDP Short-Circuit in `__init__`

When `use_megatron_fsdp=True`, `DistributedOptimizer.__init__()` returns early
(line 543) without setting up buffer ranges, gbuf mappings, or shard slicing.
Megatron FSDP manages weight/gradient memory directly.

### 5.2 `state_dict()` — FSDP Branch

```python
def state_dict(self):
    if self.ddp_config.use_megatron_fsdp:
        return self.optimizer.state_dict()
    # ... existing non-FSDP logic (strips parameter state) ...
```

For Megatron FSDP, we return the **full** inner optimizer state dict because:
- FSDP manages parameter state as DTensors (no separate `save_parameter_state`).
- PyTorch's `get_state_dict()` expects `optimizer.state_dict()` to include state.
- The `sharded_param_state_fsdp_dtensor()` path handles the Megatron-specific
  key remapping separately.

**Why not strip `"state"` like the non-FSDP path?** The legacy non-FSDP path
strips the `"state"` key from `state_dict()` and stores optimizer parameter
states (exp_avg, exp_avg_sq) in a separate `param_state` checkpoint file. This
is necessary because the non-FSDP path manages parameters in contiguous gradient
buffers and needs manual sharding logic. For Megatron FSDP, the optimizer state
is managed directly by `self.optimizer` (a standard Torch AdamW). DCP handles
the DTensor-based sharding natively. Splitting into `state_dict()` + `param_state`
adds unnecessary complexity.

### 5.3 `load_state_dict()` — FSDP Branch

```python
if self.ddp_config.use_megatron_fsdp:
    if "param_to_group_meta" in state_dict:
        state_dict["param_groups"] = self._param2group_meta_to_param_groups(
            state_dict["param_to_group_meta"], self.optimizer.param_groups
        )
        del state_dict["param_to_group_meta"]
    self.optimizer.load_state_dict(state_dict)
    return
```

Converts name-based `param_to_group_meta` back to tensor-based `param_groups`,
then delegates to the inner optimizer.

### 5.4 `sharded_state_dict()` — FSDP Branch

Only `sharding_type="fsdp_dtensor"` is supported. Calls
`sharded_param_state_fsdp_dtensor()` which:
1. Optionally initializes optimizer states with dummy values (for loading).
2. Maps tensor keys to parameter name strings via `_param_name()`.
3. Returns `{"state": ..., "param_to_group_meta": ...}`.

**Why this works for Megatron FSDP2:** The v2 path uses the same `self.optimizer`
(a standard Torch optimizer like AdamW) and the same `_param_name` mapping.
DTensor parameters are correctly identified by name. The state dict keys are the
same as the checkpoint's model state dict keys, so DCP can match them.

### 5.5 `sharding_type="fsdp_dtensor"` Rationale

The `fsdp_dtensor` checkpoint format signals to MCore's `checkpointing.py` to:
1. Use DCP (`torch.distributed.checkpoint`) for all I/O
2. Store parameters as DTensors (each DTensor carries its own sharding metadata)
3. Store optimizer state as a flat `{param_name: state}` dict

This format is the only one compatible with Megatron FSDP because the non-DCP
formats (`torch`, `torch_dist`) use gather/scatter patterns that assume contiguous
gradient buffers, which don't exist in the FSDP path.

---

## 6. Key Differences from Legacy Path

| Aspect | Legacy Megatron FSDP | Megatron FSDP2 |
|--------|---------------------|----------------|
| Parameter representation | `MegatronFSDP`-managed DTensors | Native `FSDPModule` DTensors |
| Model `state_dict_for_save_checkpoint` | `model.state_dict()` with `state_dict_pre_hook` | `model.state_dict()` (already DTensors) |
| Optimizer buffer management | Megatron FSDP managed | Standard Torch optimizer managed |
| Gradient buffer | Megatron FSDP `param_and_grad_buffer` | None (Megatron FSDP2 handles internally) |
| Load model state dict | `module.load_state_dict(custom, strict)` with `_load_state_dict_post_hook` | `super().load_state_dict(state_dict, strict)` |
| Zero gradient | `model_chunk.zero_grad_buffer()` | `model_chunk._zero_grad_buffer()` |

---

## 7. Online Checkpoint Conversion

### 7.1 Problem

Users may have checkpoints from legacy Megatron formats (ND-parallel with
`DistributedOptimizer`, or Megatron FSDP v1 baseline) and want to load them into
a Megatron FSDP2 model. The key structures differ:

| Format | Model Keys | Optimizer Keys |
|--------|-----------|----------------|
| ND-parallel | `module.layer.weight` | Tensor-keyed (by param tensor id) |
| Megatron FSDP v1 baseline | `module.layer.weight` | Tensor-keyed |
| Megatron FSDP2 | `module.layer.weight` | String-keyed (by param name) |

### 7.2 Solution: Key Mapping via `get_state_dict`

The `test_checkpoint_online_convert.py` test implements this pattern:

```python
# 1. Save source checkpoint
source_model, source_sd, source_optim = _training_loop(...)
dcp_save({"model": source_sd}, checkpoint_id=ckpt_dir)

# 2. Init target Megatron FSDP2 model
v2_model_chunks, v2_optim = _init_model_and_optimizer(...)
v2_model = _get_model_from_chunks(v2_model_chunks)

# 3. Build key mapping (canonical names match across formats)
v2_sd, _ = get_state_dict(v2_model, v2_optim)
mapped_sd = _build_key_mapping(source_sd, v2_sd)

# 4. DCP load with mapping
dcp_load(state_dict=mapped_sd, checkpoint_id=ckpt_dir)
v2_model.load_state_dict(v2_sd, strict=False)
```

The `_build_key_mapping` function strips `module.` prefixes to get canonical
parameter names, then creates a mapping from source keys to target DTensor
objects. DCP's `load` fills the target DTensors with data from the checkpoint.

### 7.3 `get_state_dict` for Megatron FSDP2

The `get_state_dict()` function in `uneven_dtensor.py` wraps PyTorch's native
`get_state_dict` with uneven DTensor preprocessing:

```python
def get_state_dict(model, optimizers, *, submodules=None, options=None):
    # Assert all params are DTensors (FSDP-wrapped)
    for param in model.parameters():
        assert isinstance(param, DTensor)

    model_state_dict, optimizer_state_dict = _get_state_dict(
        model=model, optimizers=optimizers, submodules=submodules, options=options
    )
    preprocess_state_dict_for_uneven_dtensor(model_state_dict)
    preprocess_state_dict_for_uneven_dtensor(optimizer_state_dict)
    return model_state_dict, optimizer_state_dict
```

This requires `DistributedOptimizer.state_dict()` to return a proper state dict
(the FSDP branch described in Section 5.2).

---

## 8. Uneven DTensor Handling

### 8.1 Problem

When a parameter's size is not evenly divisible by the DP world size, each rank
holds a different-sized local shard. Standard DCP assumes uniform shard sizes.

### 8.2 Solution: Chunk Metadata Patching

`preprocess_state_dict_for_uneven_dtensor()` walks the state dict, finds all
`DTensor` values, and calls `update_uneven_dtensor_chunk_metadata()` on each.
This function:

1. Gathers chunk metadata from all ranks via `all_gather_object`.
2. Computes global offsets and sizes for each rank's shard.
3. Patches the DTensor with `__create_chunk_list__` and `__create_write_items__`
   closures that DCP uses for serialization.

This is called on **both** model and optimizer state dicts.

---

## 9. DTensor Attribute Propagation

When loading a checkpoint into a Megatron FSDP2 model, parameters are DTensors.
Certain attributes (e.g., `is_embedding_parameter`, `allreduce`) set on the
original `nn.Parameter` objects by upstream layers (e.g., TE) must be propagated
to the DTensor wrappers. This is handled in `mcore_fsdp_adapter.py` at lines
307-331. Missing attributes cause optimizer misclassification (`_get_param_groups`)
and wrong gradient scaling.

---

## 10. Implementation Checklist

- [x] `DistributedOptimizer.__init__`: Add `use_fully_shard_api` guard to early return
- [x] `DistributedOptimizer.state_dict()`: Add FSDP branch to return inner optimizer state dict
- [x] `DistributedOptimizer.load_state_dict()`: Add FSDP branch for direct load
- [x] `DistributedOptimizer.sharded_state_dict()`: Add FSDP guard, route to `fsdp_dtensor`
- [x] `DistributedOptimizer.sharded_param_state_fsdp_dtensor()`: Update assertion
- [ ] `FullyShardedDataParallel._init_with_fully_shard()`: Wire `state_dict_for_save_checkpoint` to `module.state_dict()`
- [ ] End-to-end test: Save checkpoint from Megatron FSDP2, load into Megatron FSDP2 (round-trip)
- [ ] Integration test: Load legacy `fsdp_dtensor` checkpoint into Megatron FSDP2

---

## 11. Testing Strategy

### 11.1 Unit Tests

| Test | File | Status |
|------|------|--------|
| Online convert: ND-parallel → Megatron FSDP2 | `test_checkpoint_online_convert.py` | Active |
| Online convert: FSDP v1 baseline → Megatron FSDP2 | `test_checkpoint_online_convert.py` | Active |
| `get_state_dict` returns dicts | `test_fully_shard.py` | Skipped (hangs) |
| `get_state_dict` nested FSDP | `test_fully_shard.py` | Skipped (hangs) |
| `get_state_dict` strict DTensor assert | `test_fully_shard.py` | Active |
| SWiGLU/expert key transforms | `test_fsdp_dtensor_checkpoint.py` | Active |

### 11.2 Integration Tests

| Test | File | Description |
|------|------|-------------|
| Megatron FSDP2 end-to-end training | `test_mcore_fully_sharded_data_parallel.py` | Compares FSDP loss vs reference |
| ND-parallel Megatron FSDP2 | `test_mcore_nd_parallel.py` | Multi-dimensional parallelism |

### 11.3 Test Gaps

- Optimizer state dict round-trip (save + load + verify values) for Megatron FSDP2.
- Cross-format optimizer state conversion (ND-parallel optimizer → Megatron FSDP2).
- Uneven DTensor sharding with non-divisible parameter sizes.
- Frozen parameter handling in `get_state_dict`.

---

## 12. Debugging Checklist

1. **Model state dict has DTensors?** — All parameters should be `DTensor` after
   `fully_shard()`. Check with:
   ```python
   from torch.distributed.tensor import DTensor
   for name, p in model.named_parameters():
       assert isinstance(p, DTensor), f"{name}: expected DTensor, got {type(p)}"
   ```

2. **Checkpoint save succeeds?** — DCP writes to disk. Check for
   `FileSystemWriter` or `FileSystemWriterAsync` in logs.

3. **Checkpoint load succeeds?** — Verify that `dcp_load()` returns without
   errors. Set `strict_fsdp_dtensor_load=True` for strict key matching.

4. **Model parameters match after load?** — Compare source and loaded state
   dicts using `_state_dict_to_full_tensor` helper (gathers DTensors to full
   tensors for comparison).

5. **Optimizer state restored?** — Check that `optimizer.state` is non-empty
   and contains expected keys (`exp_avg`, `exp_avg_sq`, `step`).

6. **NaN after load?** — Common causes:
   - Missing `allreduce` attribute propagation (expert params misclassified)
   - Missing `overwrite_main_grad=True` for wgrad fusion (gradient doubling)
   - Wrong `gradient_accumulation_fusion` setting

---

## 13. Future Work

### 13.1 Async Checkpoint

Integrate `FileSystemWriterAsync` for non-blocking Megatron FSDP2 checkpoint saves.

### 13.2 Cross-Topology Resharding

Support loading checkpoints saved with a different DP world size. This requires
DCP's resharding planner and proper DTensor metadata.

### 13.3 Unified `set_state_dict`

Add a `set_state_dict()` wrapper in `uneven_dtensor.py` that handles the
preprocessing inverse (e.g., removing chunk metadata before applying state).

### 13.4 Complete Skipped Tests

Debug and unskip the `test_fully_shard.py` checkpoint tests that currently hang.
