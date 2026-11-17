# MFSDPv2 pipeline and heterogeneous hybrid sharding

This change enables proxy model training with pipeline parallelism, per-token loss,
and different dense/expert optimizer-instance topologies.

## Pipeline lifecycle

MFSDPv2 owns its gradient reductions. The training loop must not install the legacy
no-op grad_sync_func, which otherwise prevents non-first pipeline stages from leaving
no_sync on their last microbatch. Both inner and outer axes then finish reduction.
ExecutionOrderView.seek tolerates interleaved demand calls while rejecting events absent
from the trace. Multiple changed Partial axes reduce from inner to outer before writing
into the requested output. Per-token loss no longer fails a conservative PP check.

## Dense and expert topology

When optimizer instances do not divide expert DP, experts use one flat mesh while the
dense DP-with-CP domain retains its own outer group. Existing divisible cases keep their
hierarchical expert groups. This fallback is limited to MFSDPv2.
A local pipeline chunk may contain only dense layers or embeddings despite global EP;
validation rejects unsupported expert parameters, not an ordinary dense-only chunk.

## Validation

The dense-chunk regression covers both accepted ordinary parameters and rejected expert
parameters without a supported MoE container. Run in the supported TE/CUDA environment:

```bash
python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_validation.py
```

Historical source-branch proxy integration covered pipeline interleaving and hybrid
sharding. Those results are provenance, not a rerun of this reconstructed branch.
The combined branch is validated with new single-node FSDP and DDP runs after assembly.

Reconstructed branch check: the two dense-chunk regression cases passed.
