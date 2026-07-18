# Megatron-FSDP v2 Prototype Examples

This directory contains experimental examples for Megatron-FSDP v2. Each
example is self-contained in its own directory with setup and launch
instructions.

These examples exercise prototype APIs and may change as Megatron-FSDP v2
evolves. For the established Megatron-FSDP training and checkpoint-conversion
examples, see [`examples/megatron_fsdp`](../megatron_fsdp/README.md).

## Examples

| Example | Description |
| --- | --- |
| [`fsdp_toy`](fsdp_toy/README.md) | Standalone comparison of PyTorch FSDP2 and Megatron-FSDP v2, including CUDA graphs, HSDP, activation checkpointing, distributed checkpoints, and convergence checks. |
| [`qwen3_30b_a3b_mxfp8`](qwen3_30b_a3b_mxfp8/README.md) | Two-node Qwen3-30B-A3B training recipe using Megatron-FSDP v2, MXFP8, and Weights & Biases. |
| [`diffusers_qwenimage`](diffusers_qwenimage/README.md) | Diffusers QwenImage benchmark comparing PyTorch FSDP1 with Megatron-FSDP backends. |

Unless an example README says otherwise, run commands from the root of the
Megatron-LM repository.

## Validated case studies

The example READMEs include reproducible performance and convergence results
collected on GB200 GPUs at commit `6791bfacb9ed`:

- [`fsdp_toy`](fsdp_toy/README.md#case-study-results) compares PyTorch FSDP2,
  Megatron-FSDP v2, the v2 CUDA-graph/trace-pool path, and HSDP.
- [`diffusers_qwenimage`](diffusers_qwenimage/README.md#benchmarks) compares
  PyTorch FSDP1 with Megatron-FSDP v2 on a pretrained 60-block diffusion
  transformer and verifies fixed-batch flow-matching convergence.
- [`qwen3_30b_a3b_mxfp8`](qwen3_30b_a3b_mxfp8/README.md) covers the production
  Megatron training loop and MXFP8 parameter-gather capability.
