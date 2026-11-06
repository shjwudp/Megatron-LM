# Proxy model feature index

The reconstructed history has seven local feature commits and five independent upstream
PR commits. Each row points to its consolidated documentation; detailed raw experiment
records and training launchers remain outside this source history.

| Feature | Consolidated guide |
| --- | --- |
| MFSDPv2 pipeline/hybrid topology | [Parallelism](proxy_fsdp_parallelism.md) |
| Muon and scalar optimizers | [Optimizer](proxy_muon_optimizer.md) |
| Packed MTP and chunk lifetimes | [Pipeline](proxy_packed_mtp.md) |
| Selective recompute | [Recompute](proxy_selective_recompute.md) |
| Deterministic MoE benchmark balance | [Balance](proxy_moe_balance.md) |
| Event Tracker, NVTX, local dumps, FLOPs and analysis | [Observability](proxy_observability.md) |
| Upstream #8684 | [Transfer overlap](mfsdp_v2/pr_8684_muon_overlap.md) |
| Upstream #8782 | [Momentum stream](mfsdp_v2/pr_8782_muon_side_stream.md) |
| Upstream #8856 | [Gradient lifetime](mfsdp_v2/pr_8856_gradient_input_lifetime.md) |
| Upstream #8918 | [UB initialization](mfsdp_v2/pr_8918_native_nccl_ub.md) |
| Upstream #8950 | [All-gather registration](mfsdp_v2/pr_8950_allgather_input_registration.md) |
| Proxy memory calculator | [Calculator](proxy_memory_calculator.md) |

[Combined branch validation](proxy_validation.md) records the new eight-GPU checks.
