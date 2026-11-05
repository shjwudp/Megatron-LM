# PR #8856: Compute-stream gradient reduction inputs

Source: https://github.com/NVIDIA/Megatron-LM/pull/8856; imported upstream revision `cdf7ea1c388e80fbe0658f8cd1286955f7cbe246`.
This PR remains one independent commit, with original author and provenance trailers.

## Integration

Allocate and pack partial gradients on the compute stream. The context retains inputs until reduction completion, bounding pending module batches. Recorded mode captures forward retirement positions and reconciles the plan across DP ranks. Local packed-MTP root completion and observation hooks remain intact.

## Verification and limits

test_context.py, test_grad_buffer_lifetime.py, test_grad_buffer_retirement_consensus.py and test_parameter.py cover eager/graph lifetimes, delayed reductions, rank consensus and repeated gradient contributions. Original numerical tolerances are retained.
Run these files through torch.distributed.run with eight processes and --experimental
in the supported TE/CUDA environment. Repository pre-commit and syntax checks apply
at this reconstructed commit. Historical source-branch regressions passed, but are not
represented as newly rerun tests. New single-node proxy FSDP/DDP runs validate the final
assembled stack; a run with UB disabled does not independently exercise UB registration.
