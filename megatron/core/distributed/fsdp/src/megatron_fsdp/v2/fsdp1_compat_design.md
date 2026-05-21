# Design: FSDP1-Compatible API for Megatron FSDP2

> **Status: Experimental**
>
> This API is experimental and subject to change. It is provided for early
> evaluation and feedback purposes. Do not rely on it for production workloads.

## Motivation

Many training frameworks (e.g., Bagel, HuggingFace, LLaMA-Factory) rely on PyTorch's
FSDP1 API (`torch.distributed.fsdp.FullyShardedDataParallel`). Megatron-LM's FSDP2
implementation offers superior performance through custom buffer management, trace-based
memory pooling, FP8 support, and communication/computation overlap.

This design provides a **drop-in replacement** class that accepts the same constructor
arguments as PyTorch FSDP1 but uses Megatron FSDP2's `fully_shard()` as the backend.
This enables projects to adopt Megatron's optimized FSDP implementation with minimal
code changes.

## Goals

1. **API Compatibility**: Accept the same constructor signature as `torch.distributed.fsdp.FullyShardedDataParallel`
2. **State Dict Compatibility**: Support `FSDP.state_dict_type()` context manager and `FullStateDictConfig`
3. **Transparent Sharding**: Auto-wrap submodules using `auto_wrap_policy` and shard via `fully_shard()`
4. **Performance**: Leverage Megatron FSDP2 features (prefetch, async reduce, trace pool)

## Non-Goals

- Full feature parity with every FSDP1 option (e.g., `limit_all_gathers`, `use_orig_params=False`)
- CPU offload (Megatron FSDP2 does not currently support this)
- Backward prefetch customization (Megatron FSDP2 always prefetches)

## Architecture

```
┌──────────────────────────────────────────────────┐
│  User Code (e.g., Bagel)                         │
│  from ... import FullyShardedDataParallel as FSDP│
│  model = FSDP(model, auto_wrap_policy=..., ...)  │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│  FSDP1 Compat Layer (fsdp1_compat.py)            │
│  - Maps FSDP1 args → Megatron FSDP2 config       │
│  - Applies auto_wrap_policy to find submodules    │
│  - Calls fully_shard() on each matched submodule  │
│  - Calls fully_shard() on root module             │
│  - Provides state_dict_type() context manager     │
└──────────────────────┬───────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────┐
│  Megatron FSDP2 (v2/fully_shard.py)              │
│  - FSDPModule mixin                              │
│  - ParameterGroup → DataParallelBuffer           │
│  - BucketAllocator for memory management         │
│  - Forward/backward hooks for unshard/reshard     │
└──────────────────────────────────────────────────┘
```

## API Surface

### Constructor

```python
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp1_compat import (
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)

model = FullyShardedDataParallel(
    module,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    ),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    device_id=torch.cuda.current_device(),
    device_mesh=mesh,  # optional
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # always enabled
    cpu_offload=CPUOffload(offload_params=False),
    ignored_modules=None,
)
```

### State Dict

```python
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp1_compat import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

# Gather full state dict to rank 0
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
):
    state_dict = model.state_dict()

# Sharded state dict (DCP compatible)
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
```

## Mapping: FSDP1 → Megatron FSDP2

| FSDP1 Concept | Megatron FSDP2 Equivalent |
|---------------|---------------------------|
| `ShardingStrategy.FULL_SHARD` | `sharding_strategy="optim_grads_params"` (ZeRO-3) |
| `ShardingStrategy.SHARD_GRAD_OP` | `sharding_strategy="optim_grads"` (ZeRO-2) |
| `ShardingStrategy.NO_SHARD` | `sharding_strategy="no_shard"` (DDP) |
| `ShardingStrategy.HYBRID_SHARD` | 2D mesh `(replicate, shard)` + ZeRO-3 |
| `ShardingStrategy._HYBRID_SHARD_ZERO2` | 2D mesh + ZeRO-2 |
| `MixedPrecision.param_dtype` | `FullyShardMixedPrecisionPolicy` model dtype |
| `MixedPrecision.reduce_dtype` | `FullyShardMixedPrecisionPolicy.grad_comm_dtype` |
| `auto_wrap_policy` | Iterate submodules, call `fully_shard()` per match |
| `BackwardPrefetch.BACKWARD_PRE` | `enable_unshard_prefetch=True` (default) |
| `BackwardPrefetch.BACKWARD_POST` | `enable_unshard_prefetch=True` (same behavior) |
| `CPUOffload` | Not supported (raises warning) |

## State Dict Strategy

For `FULL_STATE_DICT`:
- All-gather each DTensor parameter to produce the full (unsharded) tensor
- If `rank0_only=True`, only rank 0 retains the gathered tensor; other ranks get `{}`
- If `offload_to_cpu=True`, move gathered tensors to CPU before returning

For `SHARDED_STATE_DICT`:
- Return the raw DTensor parameters as-is (compatible with `torch.distributed.checkpoint`)

For `LOCAL_STATE_DICT`:
- Return the local shard of each parameter (the `_local_tensor` of each DTensor)

## Auto-Wrap Policy Handling

The `auto_wrap_policy` callable is invoked on each submodule to determine which
submodules should be individually sharded. The compat layer:

1. Traverses the module tree bottom-up
2. For each submodule that matches the policy, calls `fully_shard(submodule, mesh=mesh, ...)`
3. Finally calls `fully_shard(root_module, mesh=mesh, ...)` on the root

This matches FSDP1's behavior where each wrapped submodule becomes an independent
FSDP unit with its own communication group.

## Limitations

1. **CPU Offload**: Not supported. A warning is logged if `cpu_offload=True`.
2. **`use_orig_params=False`**: Megatron FSDP2 always uses original params (DTensor views).
3. **Sync module states**: Not directly supported; broadcast happens during materialization.
4. **Rate limiter**: `limit_all_gathers` is not supported; prefetch handles this.
5. **FSDP1 internal handles**: `_get_fsdp_handles()` is not compatible; EMA workflows
   need adaptation (see example below).

## EMA Model Support

For EMA models (like Bagel's `fsdp_ema_update`), users should iterate over named
parameters directly instead of using `_get_fsdp_handles`:

```python
@torch.no_grad()
def ema_update(ema_model, model, decay=0.9999):
    for (_, ema_p), (_, model_p) in zip(
        ema_model.named_parameters(),
        model.named_parameters(),
    ):
        ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
```

## File Layout

```
megatron/core/distributed/fsdp/src/megatron_fsdp/v2/
├── fsdp1_compat.py          # <-- NEW: FSDP1-compatible wrapper
├── fsdp1_compat_design.md   # <-- NEW: This design doc
├── fully_shard.py           # Existing: fully_shard() API
├── fsdp_module.py           # Existing: FSDPModule mixin
├── param_group.py           # Existing: ParameterGroup
├── dp_buffer.py             # Existing: DataParallelBuffer
└── ...

examples/megatron_fsdp/
├── fsdp1_compat_example.py  # <-- NEW: Usage example
└── fsdp_toy.py              # Existing: fully_shard() example
```
