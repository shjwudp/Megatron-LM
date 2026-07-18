# Diffusers QwenImage: FSDP1 vs Megatron-FSDP

Minimal repro comparing PyTorch FSDP1 against Megatron-FSDP on
`QwenImageTransformer2DModel`.  All code is self-contained in `test_qwenimage.py`;
only stock packages are required below.

Run the commands below from the example directory:

```bash
cd examples/megatron_fsdp_v2_prototype/diffusers_qwenimage
```

## Environment Setup (run once)

```bash
pip install "diffusers>=0.37.0"           # QwenImage model
pip install megatron-fsdp                 # if not installed in repo
pip install huggingface_hub               # model download

# Flash attention (pick one tier):
#   Tier 1: FA3 — best perf, install from PyPI
pip install flash_attn_interface
#   Tier 2: FA2 — if FA3 unavailable
pip install flash-attn --no-build-isolation
#   Tier 3: no flash-attn at all → use --attention native below
```

## Download Model (run once)

```bash
hf download Qwen/Qwen-Image \
  --include "transformer/*" \
  --local-dir /tmp/qwen-image
```

## Run

### Single node, 4 GPU (full shard)

```bash
# Tier 1: FA3 (_flash_3)
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3

# Tier 2: FA2 (flash) — if FA3 unavailable
nsys profile \
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --bench_steps 3 --warmup_steps 1

# Tier 3: native attention — no flash-attn needed
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention native --compile --bench_steps 20 --warmup_steps 3

# PyTorch FSDP1 for comparison (swap --backend)
nsys profile \
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend fsdp1 --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --bench_steps 20 --warmup_steps 3
```

### Single node, 8 GPU (hybrid shard)

```bash
torchrun --nproc_per_node=8 test_qwenimage.py \
  --backend mfsdp --sharding hybrid \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3
```

### With numerical verification (adds sync overhead)

```bash
torchrun --nproc_per_node=4 test_qwenimage.py \
  --backend mfsdp --sharding full \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --num_gpus_per_node 4 --batch_size 4 --height 512 --width 512 \
  --attention flash --compile --verify
```

### Multi-node (2+ nodes)

```bash
torchrun --nnodes=$NNODES --node_rank=$NODE_RANK \
  --nproc_per_node=8 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  test_qwenimage.py \
  --backend mfsdp --sharding hybrid \
  --pretrained_model_name_or_path /tmp/qwen-image \
  --batch_size 4 --height 512 --width 512 \
  --attention _flash_3 --compile --bench_steps 20 --warmup_steps 3
```

## Benchmarks

Fresh results were collected on 2026-07-17 at Megatron-LM commit
`6791bfacb9ed` (job `20260717-232029-4b48`). All cases ran sequentially on the
same 4-GPU GB200 node with the pretrained 60-block QwenImage transformer, BF16,
batch size 4 per rank, 512×512 images, text length 388, full sharding,
`torch.compile`, and FA2. Results use 3 warmup and 20 measured complete training
steps. `mfsdpv2+cg` adds `--cuda-graph --trace-pool`.

| Backend | Average step | Global samples/s | Peak memory |
| --- | ---: | ---: | ---: |
| PyTorch FSDP1 | 500.31 ms | 31.98 | 75.39 GB |
| Megatron-FSDP v2 | 655.13 ms | 24.42 | **74.66 GB** |
| Megatron-FSDP v2 + CUDA graph + trace pool | **404.57 ms** | **39.55** | 86.70 GB |

The eager Megatron-FSDP v2 path uses 0.73 GB less peak memory than FSDP1 but is
slower for this workload. CUDA graph capture and trace-pool allocation change
the result materially: throughput improves by 61.9% over eager Megatron-FSDP
v2 and by 23.7% over PyTorch FSDP1, at the cost of 11.31 GB more peak memory
than FSDP1.

### Nsight Systems comparison

The eager result was repeated as two isolated Nsight Systems jobs using the
same node type and arguments (`--compile`, FA2, 3 warmup steps). The profiler
was gated by `--cuda_profiler_capture`, so model loading, compilation, and
warmup are outside the capture; each report contains 3 complete training
steps. Jobs `20260718-084609-bed6` (FSDP1) and `20260718-085130-1fbf`
(Megatron-FSDP v2) both completed successfully.

| Profile metric | PyTorch FSDP1 | Megatron-FSDP v2 | v2 - FSDP1 |
| --- | ---: | ---: | ---: |
| Profiled step | 714.64 ms | 818.69 ms | +104.05 ms (+14.6%) |
| Forward NVTX range | 255.45 ms | **243.12 ms** | -12.33 ms |
| Backward NVTX range | **427.79 ms** | 491.62 ms | +63.84 ms |
| Optimizer NVTX range | **5.26 ms** | 16.84 ms | +11.58 ms |
| All-gather GPU time / rank-step | 267.09 ms | **260.72 ms** | -6.37 ms |
| Reduce-scatter GPU time / rank-step | **190.43 ms** | 360.31 ms | +169.89 ms |
| `cudaLaunchKernel` calls / rank-step | **1,517** | 2,355 | +55.3% |

The reports were exported to SQLite to distinguish NCCL payload time from
cross-rank arrival skew. Both jobs ran on the same physical node. The apparent
reduce-scatter regression is primarily **cavitation**, not lower NCCL
bandwidth:

| Reduce-scatter timeline metric | PyTorch FSDP1 | Megatron-FSDP v2 |
| --- | ---: | ---: |
| Calls / rank-step | 61 | 61 |
| Mean cross-rank start skew / call | 3.868 ms | 6.744 ms |
| Mean cross-rank finish skew / call | 0.041 ms | 0.021 ms |
| Mean fastest-rank kernel duration | 0.904 ms | **0.868 ms** |
| Mean slowest-rank kernel duration | 4.777 ms | 7.596 ms |
| GPU 3 idle inside the RS phase / step | 89.59 ms | 217.67 ms |
| GPU 3 gaps over 1 ms / step | 0 | 60.7 |

In v2, ranks 0–2 enter each collective almost together, while rank 3 enters
6.744 ms later on average. All four ranks then finish within 0.021 ms. The
early NCCL kernels therefore spend most of their reported 7.6 ms resident and
waiting for the pacing rank; the last rank performs the actual transfer in
about 0.87 ms. This is why summing NCCL kernel durations reports 360.31
ms/rank-step even though the transfer itself is not slower than FSDP1.

The pacing-rank gaps map directly to eager gradient staging. Megatron-FSDP v2
launches 1,022 BF16 `param.grad`-to-`main_grad` foreach-copy kernels per step
(23.3 ms of GPU work); FSDP1 has none of this kernel signature. Across the
three v2 steps, 180 of the 182 GPU-3 gaps longer than 1 ms end at one of those
copy kernels—exactly 60 per step. Their mean gap is 2.761 ms, matching the
2.675 ms mean CPU `MFSDP reduce_grad` range on the pacing rank. Ranks 0–2 queue
the same copies 33–51 ms ahead of GPU execution, but rank 3 queues them only
0.143 ms ahead, so the per-module Python scan and copy launch become exposed
and repeatedly drain that GPU's work queue.

The primary optimization target is therefore the eager grad-staging path:
accumulate directly into `main_grad` (or otherwise avoid the per-parameter
copies), cache the per-module staging plan instead of rescanning parameters,
or stage the copies consistently on the reduce-scatter stream. A higher-
priority communication stream and explicit CPU/NUMA rank binding are useful
follow-up mitigations, but do not remove the copy/launch work. The strong CUDA
graph result is consistent with eliminating most of this host launch jitter.

The raw reports are named `qwenimage-fsdp1.nsys-rep` and
`qwenimage-mfsdpv2.nsys-rep`; matching `*.sqlite` exports contain the event
timelines, and `*.stats.txt` files contain the `cuda_gpu_kern_sum`,
`cuda_api_sum`, and `nvtx_sum` tables. Nsight overhead is material, so the
20-step non-profiled table above remains the throughput measurement.

### Convergence verification

Job `20260717-233136-efb3` ran 50 measured steps with `--real-data --lr 1e-5`.
Here `--real-data` means the example's deterministic fixed flow-matching batch
(`v = x1 - x0`), which is intentionally overfit to verify optimizer and
gradient correctness; it is not an external image dataset. All cases passed
the predefined `final_loss < 0.95 * initial_loss` assertion.

| Backend | Initial loss | Final loss | Final / initial | Result |
| --- | ---: | ---: | ---: | --- |
| PyTorch FSDP1 | 9.5744 | 4.7141 | 0.492 | Pass |
| Megatron-FSDP v2 | 9.5747 | 4.7814 | 0.499 | Pass |
| Megatron-FSDP v2 + CUDA graph + trace pool | 9.5740 | 4.6584 | 0.487 | Pass |

The three final/initial ratios agree within 0.012, providing a convergence
cross-check in addition to the performance comparison.

## torch FSDP1 (reference API)

```python
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("replicate", "shard"))
mp = MixedPrecision(param_dtype=bf16, reduce_dtype=bf16, buffer_dtype=bf16)
FSDP(model, device_mesh=mesh, sharding_strategy=ShardingStrategy.HYBRID_SHARD,
     mixed_precision=mp,
     auto_wrap_policy=ModuleWrapPolicy({QwenImageTransformerBlock}),
     backward_prefetch=BackwardPrefetch.BACKWARD_PRE, forward_prefetch=True,
     use_orig_params=True, limit_all_gathers=True)
```

## Megatron-FSDP (reference API)

```python
mesh = init_device_mesh("cuda", (num_nodes, gpus_per_node),
                        mesh_dim_names=("dp_outer", "dp_shard"))
mesh[("dp_outer", "dp_shard")]._flatten("hsdp")
mp = MixedPrecisionPolicy(main_params_dtype=bf16, main_grads_dtype=bf16,
                          grad_comm_dtype=bf16)
fully_shard_model(
    module=model, fsdp_unit_modules=[QwenImageTransformerBlock],
    device_mesh=mesh, dp_shard_dim="dp_shard",
    dp_outer_dim="dp_outer", hybrid_fsdp_group=mesh["hsdp"].get_group(),
    zero_dp_strategy=3,                # ZeRO-3 within node
    outer_dp_sharding_strategy=0,      # REPLICATE across nodes (match FSDP1 HYBRID_SHARD)
    sync_model_each_microbatch=True,
    overlap_grad_reduce=True, overlap_param_gather=True,
    mixed_precision_policy=mp,
)

# Optimizer
fully_shard_optimizer(optim)
optim.step(sync_grad_before_optimizer_step=True, install_optimized_model_weights=True)
optim.zero_grad(set_to_none=True, zero_grad_buffer=True)
```

## Notes

- Per-block `torch.compile` (identical for both backends):
  ```python
  for blk in model.module.transformer_blocks:
      blk.compile()
  ```
- `--verify` gradient probe — mfsdp does not expose gradients via `param.grad`:
  ```python
  def _local_grad(p):
      g = p.grad
      if g is None and hasattr(p, "get_main_grad"): g = p.get_main_grad()
      if g is None: g = getattr(p, "main_grad", None)
      if g is None: g = getattr(p, "decoupled_grad", None)
      if hasattr(g, "to_local"): g = g.to_local()
      return g
  ```
- FA3 autograd shim is applied inline by the script — no manual patching required.
- `hybrid` sharding on a single node falls back to `full` shard automatically.
