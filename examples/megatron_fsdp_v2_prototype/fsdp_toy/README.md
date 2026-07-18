# Megatron-FSDP v2 Toy Example

`fsdp_toy.py` is a standalone distributed training example for comparing
PyTorch FSDP2 with Megatron-FSDP v2. It uses a small MLP-style model and does
not depend on the Megatron training loop.

The example covers:

- per-layer and root-module sharding with `fully_shard()`;
- CUDA graph capture and trace-pool allocation;
- activation checkpointing;
- HSDP with outer optimizer-state sharding;
- distributed checkpoint save and resume;
- CUDA memory snapshots; and
- deterministic teacher-student convergence verification.

## Run

Run these commands from the Megatron-LM repository root.

PyTorch FSDP2 baseline:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4
```

Megatron-FSDP v2:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4 \
  --use-megatron-fsdp
```

Enable CUDA graphs, trace-pool allocation, and activation checkpointing:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4 \
  --use-megatron-fsdp --cuda-graph --use-trace-pool \
  --activation-checkpoint
```

Enable the deterministic convergence check and distributed checkpoints:

```bash
torchrun --standalone --nproc_per_node=2 \
  examples/megatron_fsdp_v2_prototype/fsdp_toy/fsdp_toy.py \
  --model-dim 512 --n-layers 2 --batch-size 4 \
  --use-megatron-fsdp --use-real-data \
  --ckpt-dir /tmp/mfsdp-v2-toy-checkpoints
```

## Selected options

| Option | Default | Description |
| --- | --- | --- |
| `--model-dim` | `1024` | Model hidden dimension. |
| `--n-layers` | `3` | Number of toy transformer-style blocks. |
| `--use-megatron-fsdp` | off | Use Megatron-FSDP v2 instead of PyTorch FSDP2. |
| `--cuda-graph` | off | Capture Megatron-FSDP layer execution in CUDA graphs. |
| `--use-trace-pool` | off | Use the trace-pool allocator for stable buffer addresses. |
| `--activation-checkpoint` | off | Recompute block activations during backward. |
| `--enable-hsdp` | off | Use a `2 x N` HSDP mesh; requires an even world size and Megatron-FSDP. |
| `--release-memory-pool` | off | Release Megatron-FSDP allocator slots after backward. |
| `--ckpt-dir` | unset | Save and resume distributed checkpoints in this directory. |
| `--use-real-data` | off | Use deterministic teacher-student data and assert convergence. |
| `--record-memory-history DIR` | unset | Write one CUDA memory snapshot per rank. |

Use `--help` for the complete option list.

## Case-study results

The following results were collected on 2026-07-17/18. The first three rows
use Megatron-LM commit `6791bfacb9ed` (job `20260717-231830-e872`); the HSDP
row uses the BF16 replica-refresh fix at `d3ec47ba68e6` (job
`20260718-003342-8c05`). Each case used one 4-GPU GB200 node, BF16, model
dimension 2048, 4 layers, sequence length 128, batch size 8 per rank, 20
warmup steps, and 100 measured optimizer steps. Step time covers forward,
backward, communication, and the optimizer update; all ranks are synchronized
before each measurement. Memory is the maximum across ranks.

| Backend | Average step | Samples/s | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FSDP2, full shard | 13.047 ms | 2,452.7 | 0.956 GB | 1.298 GB |
| Megatron-FSDP v2, full shard | 11.569 ms | 2,766.0 | 1.010 GB | 1.206 GB |
| Megatron-FSDP v2, CUDA graph + trace pool | **9.220 ms** | **3,470.8** | 1.323 GB | 1.837 GB |
| Megatron-FSDP v2, HSDP | 12.993 ms | 2,462.9 | 1.163 GB | 1.399 GB |

Megatron-FSDP v2 full shard is 12.8% faster than PyTorch FSDP2 in this case.
CUDA graph capture with trace-pool allocation adds another 25.5% throughput
over the Megatron-FSDP v2 eager path, with a 52.3% increase in peak reserved
memory. HSDP is included primarily as a sharding-capability check at this small
single-node scale; its throughput is within 0.5% of the PyTorch FSDP2 baseline.

### Convergence verification

Job `20260717-232603-37df` ran the deterministic teacher-student workload for
20 optimizer steps on the same 4-GPU node with model dimension 512, 2 layers,
sequence length 128, and batch size 4 per rank. The corrected full-shard/HSDP
comparison was rerun at `d3ec47ba68e6` in job `20260718-002854-45c4`. Every
mode satisfied `final_loss < 0.5 * initial_loss`.

| Backend | Initial loss | Final loss | Final / initial | Result |
| --- | ---: | ---: | ---: | --- |
| PyTorch FSDP2, full shard | 1.3577e-3 | 7.8260e-5 | 0.058 | Pass |
| Megatron-FSDP v2, full shard | 1.3577e-3 | 6.1461e-5 | 0.045 | Pass |
| Megatron-FSDP v2, CUDA graph + trace pool | 1.3577e-3 | 6.1461e-5 | 0.045 | Pass |
| Megatron-FSDP v2, HSDP | 1.3577e-3 | 6.1469e-5 | 0.045 | Pass |

The CUDA-graph path matches the eager Megatron-FSDP v2 loss endpoints in this
deterministic test. After the BF16 outer-replica refresh fix, HSDP differs from
the eager full-shard final loss by only `8e-9`; the pre-fix HSDP final loss was
`4.4203e-4`, which is what exposed the stale-replica correctness bug.
