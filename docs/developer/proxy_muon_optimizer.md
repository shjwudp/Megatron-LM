# MFSDPv2 Muon and Adam for a proxy model

The optimizer registry partitions matrix parameters into FsdpMuon and scalar/excluded
parameters into Adam (or the configured scalar optimizer), then wraps and chains each
group. CSA compressor position-bias tables are explicitly excluded from Muon even
though their storage is two-dimensional. This is a parameter-role rule, not a shape preset.

## Sharding and gradients

Owner gather/scatter supports HSDP replicated/sharded meshes and fully sharded N-D
meshes. Work assignment is grouped by mesh so dense and expert groups may differ.
DTensor-aware gradient norms avoid repeated expert counts and invalid get_group calls
on a multidimensional mesh. All ranks participate even when a local parameter shard is
empty. Compute weights refresh for all trainable parameters so empty optimizer shards
do not cause mismatched next-step collectives.

## Scope

Tensor parallel size one is the supported integration path. Split QKV/head/FC1
orthogonalization and layer-wise optimizer are not added. Momentum checkpoint round-trip
is not claimed. Later upstream transfer-overlap changes retain separate commits.

## Validation

Changed Python modules pass syntax inspection and repository pre-commit. Historical
source tests exercised optimizer classification and hybrid meshes; they are not a new
numerical run. The reconstructed stack is checked with the existing MFSDPv2 optimizer
regressions where the environment supports them and with single-node FSDP/DDP proxy
training after assembly. No production model dimensions are needed to test these rules.
