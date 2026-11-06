# Packed MTP and combined 1F1B lifetimes

This document combines packed-sequence support, standalone MTP construction and schedule
node lifetime fixes for a proxy model. Training launchers and cluster configuration are
maintained outside this feature commit.

## Data and schedule boundaries

The combined schedule carries packed sequence metadata through preprocessing, decoder
chunks and MTP. It unwraps the correct FSDP root, installs child-unit backward hooks and
finishes the root after every contributing backward segment. Reused embedding parameters
must not reduce before the MTP contribution arrives. The parameter-group regression
covers repeated contributions and lifecycle completion.

An MTP-only virtual chunk has no decoder layer from which to derive its attention spec.
Both GPT builders therefore select the configured experimental attention variant when
building that chunk, preserving CSA instead of falling back to ordinary MLA.

## Reference lifetime

ScheduleNode stores its default backward callback as a static function. An unused loss
node on an intermediate pipeline chunk can then release its chunk state immediately.
A stored bound method would create node→method→node and retain input tensors and their
gradients until Python cyclic GC. The regression disables GC and verifies that node,
input and gradient weakrefs become dead without invoking backward or explicit cleanup.
No extra collectives, CUDA synchronization or per-iteration GC are introduced.

## Validation

```bash
python -m torch.distributed.run --standalone --nproc-per-node 1 -m pytest -q \
  tests/unit_tests/training/models/test_gpt_builder.py \
  tests/unit_tests/pipeline_parallel/test_schedule_node_autograd_boundary.py
```

Packed MTP and FSDP parameter lifecycle regressions are colocated under a2a_overlap and
MFSDPv2 tests. New single-node FSDP/DDP integration after assembly covers VPP, EP overlap,
selective recompute and manual GC across a longer iteration window. Historical source
runs are not substituted for this rebuilt branch's verification.

Reconstructed branch result: GPT builder and schedule boundary suite, 78 passed.
