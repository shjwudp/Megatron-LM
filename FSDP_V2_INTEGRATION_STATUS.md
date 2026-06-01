# FSDP v2 Integration Tracker

Source: `mfsdp_refactor_ckpt` (commits since `2ee3bfb2c`)
Target: `mfsdp_nt4_dev`

## Done

| File | Method | Notes |
|------|--------|-------|
| `megatron/core/distributed/fsdp/src/megatron_fsdp/v2/` (all) | `checkout` from refactor_ckpt | ZeRO-1/2/3, MixedPrecisionPolicy rename, checkpoint sync |
| `megatron/core/distributed/fsdp/checkpoint.py` | `checkout` from refactor_ckpt | |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/uneven_dtensor.py` | `checkout` from refactor_ckpt | |
| `tools/checkpoint/checkpoint_inspector.py` | `checkout` from refactor_ckpt | Batched comparison, L2 norm output |
| `examples/megatron_fsdp/fsdp_toy.py` | `checkout` from refactor_ckpt | New file |
| `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` | `checkout` + re-apply nemotron changes | MambaLayer, EP overlap, fine-grained hooks, Mamba TP preserved |
| `megatron/core/optimizer/distrib_optimizer.py` | `git apply` patch | |
| `megatron/core/optimizer/optimizer_config.py` | `git apply` patch | Added `use_megatron_fsdp` field |
| `megatron/training/config/training_config.py` | `git apply` patch | |
| `megatron/core/distributed/distributed_data_parallel_config.py` | Manual edit | Added `use_megatron_fsdp_v2` field |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/distributed_data_parallel_config.py` | `checkout` + add nemotron field | Added `megatron_fsdp_enable_fine_grained_param_gather` |

| `megatron/training/checkpointing.py` | Manual edit | v2 imports, `_is_megatron_fsdp_v2()`, v2 branches in save/load, post-load sync |
| `megatron/training/arguments.py` | Manual edit | `--use-megatron-fsdp-v2` arg + auto-enable `use_megatron_fsdp` |
| `megatron/training/training.py` | Manual edit | Per-param norm logging in training loop |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py` | Skipped | v1 wrapper changes only — v2 path bypasses this file |

## Patch files

Generated patches stored at `/tmp/fsdp_patches/` — see individual `.patch` files for exact diffs.
