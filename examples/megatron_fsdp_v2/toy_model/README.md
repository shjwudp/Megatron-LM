# Toy Model

Standalone examples (not tied to the Megatron-LM training framework)
demonstrating Megatron-FSDP v2 usage.

## Scripts

### `fsdp_toy.py` — Convergent training with a custom MLP model

Small MLP model with teacher-student regression, checkpointing, and convergence
verification. Supports both Megatron-FSDP v2 and PyTorch FSDP2.

```bash
# Basic Megatron-FSDP v2
torchrun --nproc_per_node=2 examples/megatron_fsdp_v2/toy_model/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4 --use-megatron-fsdp

# With CUDA graph and trace pool
torchrun --nproc_per_node=2 examples/megatron_fsdp_v2/toy_model/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4 \
    --use-megatron-fsdp --cuda-graph --use-trace-pool

# Compare with PyTorch FSDP2
torchrun --nproc_per_node=2 examples/megatron_fsdp_v2/toy_model/fsdp_toy.py \
    --model-dim 512 --n-layers 2 --batch-size 4

# Convergence test (teacher-student regression)
torchrun --nproc_per_node=2 examples/megatron_fsdp_v2/toy_model/fsdp_toy.py \
    --model-dim 1024 --n-layers 3 --batch-size 8 --use-real-data --use-megatron-fsdp
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dim` | `1024` | Hidden dimension size |
| `--n-layers` | `3` | Number of MLP layers |
| `--batch-size` | `8` | Micro-batch size |
| `--seq-len` | `128` | Sequence length |
| `--epochs` | `2` | Training epochs |
| `--steps-per-epoch` | `10` | Steps per epoch |
| `--lr` | `1e-3` | Learning rate |
| `--seed` | `1234` | Random seed |
| `--use-megatron-fsdp` | off | Use Megatron-FSDP v2 instead of PyTorch FSDP2 |
| `--cuda-graph` | off | CUDA graph capture (Megatron-FSDP only) |
| `--use-trace-pool` | off | TracePoolAllocator for stable buffer addresses |
| `--activation-checkpoint` | off | Activation checkpointing |
| `--use-real-data` | off | Teacher-student regression with MSE loss |
| `--convergence-threshold` | `0.5` | Assert final_loss < initial_loss * threshold |
| `--ckpt-dir` | — | DCP checkpoint directory |
| `--ckpt-interval` | `20` | Steps between checkpoints |
| `--log-interval` | `5` | Steps between log messages |
| `--release-memory-pool` | off | Release allocator slot tensors after each backward |
| `--record-memory-history DIR` | — | Dump CUDA memory snapshots to DIR |

### `bench_llama.py` — Throughput benchmark with torchtitan Llama 3.1

Benchmark Megatron-FSDP v2 against PyTorch FSDP2 using torchtitan's Llama 3.1 model.
Reports avg ms/step, tokens/s, and peak GPU memory.

```bash
pip install torchtitan

# Megatron-FSDP v2
torchrun --nproc_per_node=8 \
  examples/megatron_fsdp_v2/toy_model/bench_llama.py \
  --backend mfsdp --flavor 8B --batch-size 1 --seq-len 8192 --bench-steps 20 --warmup-steps 5

# PyTorch FSDP2 (comparison)
torchrun --nproc_per_node=8 \
  examples/megatron_fsdp_v2/toy_model/bench_llama.py \
  --backend torchfsdp --flavor 8B --batch-size 1 --seq-len 8192 --bench-steps 20 --warmup-steps 5

# Debug OOM with memory snapshots
torchrun --nproc_per_node=8 \
  examples/megatron_fsdp_v2/toy_model/bench_llama.py \
  --backend mfsdp --flavor 8B --batch-size 1 --seq-len 8192 \
  --record-memory-history /tmp/mem_dump
```

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `mfsdp` | `mfsdp` or `torchfsdp` |
| `--flavor` | `debugmodel` | Model size: `debugmodel`, `8B`, `70B`, … |
| `--sharding-strategy` | `optim_grads_params` | `no_shard`, `optim`, `optim_grads`, `optim_grads_params` |
| `--batch-size` | `1` | Micro-batch size |
| `--seq-len` | `2048` | Sequence length |
| `--bench-steps` | `20` | Benchmark steps after warmup |
| `--warmup-steps` | `5` | Warmup steps |
| `--loss-chunk-size` | `2048` | Chunk size for memory-efficient loss |
| `--seed` | `1234` | Random seed |
| `--debug-fsdp` | off | Verbose M-FSDP v2 debug logging |
| `--record-memory-history DIR` | — | Dump CUDA memory snapshots to DIR |
| `--record-memory-history-oom-only` | off | Only dump on OOM |

## Reference API

```python
from torch.distributed.device_mesh import init_device_mesh

# Megatron-FSDP v2
from megatron_fsdp.v2 import fully_shard
from torch.distributed.fsdp import MixedPrecisionPolicy

mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, main_grads_dtype=torch.bfloat16)

for layer in model.layers.values():      # torchtitan layout
    fully_shard(layer, mesh=mesh, mp_policy=mp)
fully_shard(model, mesh=mesh, mp_policy=mp)
```
