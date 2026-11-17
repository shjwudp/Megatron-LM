# MFSDPv2 pipeline and heterogeneous hybrid sharding

MFSDPv2 supports proxy model training with pipeline parallelism, per-token loss,
and different dense and expert optimizer-instance topologies.

## Pipeline lifecycle

MFSDPv2 owns its gradient reductions. A legacy no-op grad_sync_func prevents
non-first pipeline stages from leaving no_sync on their last microbatch, so the
training loop must not install it. Both inner and outer axes then finish reduction.
ExecutionOrderView.seek supports interleaved demand calls and returns false for
events absent from the trace. When several Partial axes change, reduction proceeds
from inner to outer before writing the requested output. With per-token loss, the
adapter warns that PP>1 and multiple optimizer instances are untested.

## Dense and expert topology

When the optimizer-instance count does not divide expert DP, experts use one flat
mesh while the dense DP-with-CP domain keeps its own outer group. Divisible cases
keep hierarchical expert groups. This fallback applies only to MFSDPv2.
A pipeline chunk may contain only dense layers or embeddings even with global EP.
Validation rejects expert parameters without a supported MoE container.

## Validation

The dense-chunk regression checks that ordinary parameters are accepted and expert
parameters without a supported MoE container are rejected. Run it in a supported
TE/CUDA environment:

```bash
python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_validation.py
```
