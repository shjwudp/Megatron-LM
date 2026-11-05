# PR #8782: Muon momentum and communication streams

Source: https://github.com/NVIDIA/Megatron-LM/pull/8782; imported upstream revision `53773a00e95a953c206c55b6c2ade7203a02ea85`.
This PR remains one independent commit, with original author and provenance trailers.

## Integration

Momentum and gather run on a dedicated stream. All communicating groups submit gather early; Newton–Schulz, scatter and updates run on the calling stream. Groups requiring no communication skip owner transfers. Cached rank ranges replace lookahead. Concurrent full tensors can increase optimizer peak memory; no universal throughput improvement is claimed.

## Verification and limits

The existing Muon/gather-scatter regressions cover update numerics, separate streams, cached ranges and communication-free groups.
Run these files through torch.distributed.run with eight processes and --experimental
in the supported TE/CUDA environment. Repository pre-commit and syntax checks apply
at this reconstructed commit. Historical source-branch regressions passed, but are not
represented as newly rerun tests. New single-node proxy FSDP/DDP runs validate the final
assembled stack; a run with UB disabled does not independently exercise UB registration.
