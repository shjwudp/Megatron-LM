# Megatron-FSDP v2 Examples

Standalone examples demonstrating Megatron-FSDP v2 (`fully_shard`) with different
model types, independent of the Megatron-LM training framework.

| Directory | Description |
|-----------|-------------|
| [`toy_model/`](toy_model/) | MLP convergence test + Llama 3.1 throughput benchmark, CUDA graph, checkpointing |
| [`diffusers_qwenimage/`](diffusers_qwenimage/) | QwenImage transformer — benchmark v1/v2/FSDP1, torch.compile, memory history |

## Shared patterns

All examples use the same v2 API:

```python
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fully_shard
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cuda", (world_size,))
for layer in model.layers:          # or model.transformer_blocks
    fully_shard(layer, mesh=mesh)   # leaf FSDP unit
fully_shard(model, mesh=mesh)       # root
```

See [Megatron-FSDP User Guide](../../docs/user-guide/features/megatron_fsdp.md)
for the full feature guide and API reference.
