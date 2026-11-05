# PR #8950: Register all-gather inputs with NCCL UB

Source: https://github.com/NVIDIA/Megatron-LM/pull/8950; imported upstream revision `7843b400d75b791d3ab539bcacd17706b28346d9`.
This PR remains one independent commit, with original author and provenance trailers.

## Integration

Preserve input DBuffer identity across multi-axis all-gather. When outputs use symmetric memory, register symmetric inputs as well. Per-axis timing and the unpadded path are retained; allocation policy is unchanged. This PR depends on the native initialization fix.

## Verification and limits

test_dbuffer.py and test_quantized_dbuffer.py cover multidimensional meshes, independent symmetric input/output combinations, MXFP8 planes and CUDA Graph replay.
Run these files through torch.distributed.run with eight processes and --experimental
in the supported TE/CUDA environment. Repository pre-commit and syntax checks apply
at this reconstructed commit. Historical source-branch regressions passed, but are not
represented as newly rerun tests. New single-node proxy FSDP/DDP runs validate the final
assembled stack; a run with UB disabled does not independently exercise UB registration.
