# Diffusers QwenImage

Benchmark Megatron-FSDP v1/v2 against PyTorch FSDP1 on `QwenImageTransformer2DModel`
from diffusers. Self-contained in `test_qwenimage.py`.

## Environment Setup

```bash
pip install "diffusers>=0.37.0"
pip install megatron-fsdp
pip install huggingface_hub

# Flash attention (pick one):
pip install flash_attn_interface   # FA3 — best perf
pip install flash-attn --no-build-isolation   # FA2
# No flash-attn → use --attention native
```

## Download Model

```bash
hf download Qwen/Qwen-Image \
  --include "transformer/*" \
  --local-dir /tmp/qwen-image
```

## Run

### Single node, 4 GPU (full shard)

```bash
# Megatron-FSDP v2
torchrun --nproc_per_node=4 examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend mfsdpv2 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3

# Megatron-FSDP v2 with CUDA graph
torchrun --nproc_per_node=4 examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend mfsdpv2 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --cuda-graph --trace-pool \
  --bench_steps 20 --warmup_steps 3

# Megatron-FSDP v1
torchrun --nproc_per_node=4 examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3

# PyTorch FSDP1 (comparison)
torchrun --nproc_per_node=4 examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend fsdp1 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --bench_steps 20 --warmup_steps 3

# With numerical verification (adds sync overhead — timing invalid)
torchrun --nproc_per_node=4 examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend mfsdpv2 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --verify
```

### Single node, 8 GPU (hybrid shard)

```bash
torchrun --nproc_per_node=8 examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend mfsdpv2 --sharding hybrid \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3
```

### Multi-node (2+ nodes)

```bash
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  examples/megatron_fsdp_v2/diffusers_qwenimage/test_qwenimage.py \
  --backend mfsdpv2 --sharding hybrid \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3
```

## Benchmarks

`QwenImageTransformer2DModel`, `bs=4`, `512×512`, `bf16`, `torch.compile`, FA2.
`[mfsdpv2+cg]` uses `--cuda-graph --trace-pool`.

| Backend | 8×H100 | 4×GB200 |
|---------|--------|---------|
| **fsdp1** | 729 ms / 60.2 GB | 679 ms / 75.4 GB |
| **mfsdpv2** | 769 ms / 59.3 GB | 647 ms / 74.7 GB |
| **mfsdpv2+cg** | **674 ms** / 68.3 GB | **364 ms** / 88.7 GB |

CG delivers **11% faster** on H100 and **44% faster** on GB200 at the cost
of higher peak memory (pool-backed graph buffers).

## Reference APIs

### Megatron-FSDP v2 (`--backend mfsdpv2`)

```python
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import fully_shard, MixedPrecisionPolicy

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
mp = MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)

for blk in model.transformer_blocks:
    fully_shard(blk, mesh=mesh, mp_policy=mp, sharding_strategy="optim_grads_params",
                enable_unshard_prefetch=True, enable_async_reduce_grad=True,
                enable_trace_pool=True, enable_cuda_graph=True)
fully_shard(model, mesh=mesh, mp_policy=mp, sharding_strategy="optim_grads_params",
            enable_unshard_prefetch=True, enable_async_reduce_grad=True,
            enable_trace_pool=True)
```

### Megatron-FSDP v1 (`--backend mfsdp`)

```python
from megatron_fsdp import fully_shard_model, fully_shard_optimizer, MixedPrecisionPolicy

mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("dp_outer", "dp_shard"))
mesh[("dp_outer", "dp_shard")]._flatten("hsdp")
mp = MixedPrecisionPolicy(main_params_dtype=bf16, main_grads_dtype=bf16, grad_comm_dtype=bf16)

model = fully_shard_model(
    module=model, fsdp_unit_modules=[QwenImageTransformerBlock],
    device_mesh=mesh, dp_shard_dim="dp_shard",
    dp_outer_dim="dp_outer", hybrid_fsdp_group=mesh["hsdp"].get_group(),
    zero_dp_strategy=3, outer_dp_sharding_strategy=0,
    sync_model_each_microbatch=True,
    overlap_grad_reduce=True, overlap_param_gather=True,
    mixed_precision_policy=mp,
)
fully_shard_optimizer(optim)
optim.step(sync_grad_before_optimizer_step=True, install_optimized_model_weights=True)
optim.zero_grad(set_to_none=True, zero_grad_buffer=True)
```

### PyTorch FSDP1 (`--backend fsdp1`)

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("replicate", "shard"))
mp = MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)
FSDP(model, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
     mixed_precision=mp,
     auto_wrap_policy=ModuleWrapPolicy({QwenImageTransformerBlock}),
     backward_prefetch=BackwardPrefetch.BACKWARD_PRE, forward_prefetch=True,
     use_orig_params=True, limit_all_gathers=True)
```

## Notes

- Per-block `torch.compile` for v2 (model is not wrapped in a FSDP container):
  ```python
  for blk in model.transformer_blocks:
      blk.compile()
  ```
  For v1/fsdp1, use `model.module.transformer_blocks` instead.
- `--verify` probes global loss + grad-norm across backends. mfsdp gradients are
  accessed via `param.get_main_grad()` or `param.main_grad` — see `_local_grad()`
  in the script.
- FA3 autograd shim is applied inline. No manual patching required.
- `hybrid` sharding on a single node falls back to `full` shard.
