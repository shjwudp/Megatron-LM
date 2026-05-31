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

## Pending (blocked by nemotron reformatting)

| File | Reason | What's Missing |
|------|--------|----------------|
| `megatron/training/checkpointing.py` | Heavily reformatted by nemotron | `_is_megatron_fsdp_v2()`, `_apply_mcore_postprocess` import, v2 save/load branches, post-load sync |
| `megatron/training/arguments.py` | Context mismatch | FSDP v2 CLI args |
| `megatron/training/training.py` | Context mismatch | FSDP v2 training loop hooks |
| `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py` | Context mismatch | v1 wrapper changes (may not be needed for v2 path) |

## Nemotron-specific changes preserved in adapter

- `from megatron.core.ssm.mamba_layer import MambaLayer`
- `self.fsdp_unit_modules = [TransformerLayer, MambaLayer]` (line ~173)
- EP overlap assertions before `super().__init__()` (lines ~179-197)
- `enable_fine_grained_param_gather_hook` with EP + `megatron_fsdp_enable_fine_grained_param_gather` conditions
- `enable_fine_grained_param_gather_backward_hook` for EP
- Old-style `finish_grad_sync` (lambda waiting on `ctx.rs_stream`)
- `_detect_parallelism_type(param_name, module, param=None)` signature
- Mamba parameter-level TP detection (checks `param.tensor_model_parallel` attribute)

## Patch files

Generated patches stored at `/tmp/fsdp_patches/` — see individual `.patch` files for exact diffs.
