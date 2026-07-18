# Qwen3-30B-A3B MXFP8 Training

This example is a two-node SLURM recipe for training Qwen3-30B-A3B with
Megatron-FSDP v2, MXFP8 parameter gathering, expert parallelism, and Weights &
Biases logging.

The checked-in values target two nodes with four GB200 GPUs per node. Treat the
script as a starting point and adapt its SLURM, container, mount, data, and
tokenizer settings to your cluster.

## Configuration summary

| Setting | Value |
| --- | --- |
| Nodes / GPUs | 2 nodes, 4 GPUs per node |
| Parallelism | TP1, PP1, CP1, EP4, ETP1 |
| FSDP | Megatron-FSDP v2, `optim_grads_params` |
| Precision | BF16 training, MXFP8, FP8 parameter gather |
| Sequence length | 4096 |
| Batch size | MBS4, GBS128 |
| Dispatcher | all-to-all |
| Container | `nvcr.io/nvidia/nemo:26.04` |

## Configure

At minimum, update the `#SBATCH` account and partition, the container mounts,
and these environment variables:

```bash
export MEGATRON_PATH=/path/to/Megatron-LM
export OUTPUT_PATH=/path/to/output
export DATA_PATH=/path/to/data/c4/en/c4-train.en_6_text_document
export TOKENIZER_MODEL=/path/to/data/c4/en/tokenizer

# Optional W&B settings
export WANDB_API_KEY=...
export WANDB_ENTITY=...
```

## Launch

```bash
SCRIPT=examples/megatron_fsdp_v2_prototype/qwen3_30b_a3b_mxfp8/\
qwen3-30b-a3b.gbs128_mbs4_seq4096_n2_mfsdp2_mxfp8_wandb.sh
sbatch "$SCRIPT"
```

The script writes checkpoints, TensorBoard events, and W&B files below
`OUTPUT_PATH`.

## Case-study results

The following results were collected on 2026-07-18 at Megatron-LM commit
`6791bfacb9ed` using two GB200 nodes with four GPUs per node. All cases used
the full Qwen3-30B-A3B shape (30.53B total and 3.35B active parameters),
TP1/PP1/CP1/EP4, MBS4/GBS128, sequence length 4096, all-to-all token dispatch,
selective `moe_act` recomputation, and mock data with forced-balanced routing.
The table reports 40 measured optimizer steps after 10 warmup steps.

| Backend | Median step | Median TFLOP/s/GPU | Samples/s | Peak device memory | W&B |
| --- | ---: | ---: | ---: | ---: | --- |
| Megatron-FSDP v1, BF16 | 5,011.35 ms | 300.85 | 25.36 | 174.69 GB | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/5911d158ab53464d87968f208106793a) |
| Megatron-FSDP v2, BF16 | **4,661.90 ms** | **323.45** | **26.97** | 183.09 GB | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/acd8c6a872434775833a285c88ae1f30) |
| Megatron-FSDP v2, MXFP8 parameter gather | 6,863.15 ms | 219.70 | 18.49 | **170.85 GB** | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/9811f6c0d56640e2a0308b623b6352b9) |

Megatron-FSDP v2 BF16 is 7.5% faster than v1 BF16 in this workload, at the
cost of 8.40 GB more peak device memory. MXFP8 parameter gathering reduces
peak device memory by 12.25 GB relative to v2 BF16, but is 32.1% slower here;
it demonstrates the low-precision parameter-gather capability rather than a
throughput win for this all-to-all EP4 configuration.

The validated launch uses `CUDA_DEVICE_MAX_CONNECTIONS=8`, native
cross-entropy fusion, and selective `moe_act` recomputation. Without selective
recomputation, the BF16 v2 case reached 189.25 GB and ran out of memory after
iteration 10.

### Real-data convergence

The BF16 v1 and v2 cases were also run for 50 optimizer steps on SlimPajama
without forced-balanced routing. The model shape and parallel configuration
were unchanged from the performance matrix, and both runs completed with zero
skipped and zero NaN iterations.

| Backend | Initial train loss | Final train loss | Final validation loss | Final / initial | Skipped / NaN | W&B |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Megatron-FSDP v1, BF16 | 12.34454 | 7.690141 | 7.711185 | 0.6230 | 0 / 0 | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/97c52d949268483fbd930ad34c80f770) |
| Megatron-FSDP v2, BF16 | 12.39409 | 7.632825 | 7.651939 | 0.6159 | 0 / 0 | [run](https://wandb.ai/adlr/jianbinc-qwen3-30b-GB200-benchmark/runs/fe1917bfbfab49928324637d2c563576) |

The final train and validation losses agree within 0.8% between backends, and
both loss curves decrease normally. This real-data check validates numerical
training behavior; the forced-balanced mock-data runs above remain the
controlled performance comparison.
