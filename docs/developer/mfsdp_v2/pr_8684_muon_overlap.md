# PR #8684: Muon transfers across parameter groups

Source: https://github.com/NVIDIA/Megatron-LM/pull/8684; imported upstream revision `c5d10748c0372272a470edebf9acd8065d4f8b51`.
This PR remains one independent commit, with original author and provenance trailers.

## Integration

Gather/scatter return asynchronous Work handles. Muon posts transfers before waiting for their results and overlaps them with Newton–Schulz work. The integration retains multidimensional shard views, per-mesh owner assignment, profiling and compute-weight refresh after updates. This PR alone retains lookahead; the next PR removes it.

## Verification and limits

test_muon.py and test_gather_scatter.py compare sharded and unsharded optimizer updates and asynchronous transfers.
Run these files through torch.distributed.run with eight processes and --experimental
in the supported TE/CUDA environment. Repository pre-commit and syntax checks apply
at this reconstructed commit. Historical source-branch regressions passed, but are not
represented as newly rerun tests. New single-node proxy FSDP/DDP runs validate the final
assembled stack; a run with UB disabled does not independently exercise UB registration.
