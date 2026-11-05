# PR #8918: Native NCCL UB initialization and CLI defaults

Source: https://github.com/NVIDIA/Megatron-LM/pull/8918; imported upstream revision `9ef9176cc8a9c68197046ee9c78e592c709756e6`.
This PR remains one independent commit, with original author and provenance trailers.

## Integration

Initialize an uninitialized NCCL backend before symmetric rendezvous; reuse existing communicators on the hot path. The automatic manual-registration/double-buffer defaults of use-nccl-ub remain restricted to MFSDPv1. MFSDPv2 UB requires symmetric memory and padded shards with symmetric registration enabled.

## Verification and limits

The DBuffer symmetric reduction and test_mfsdp_ub_arguments.py regressions cover cold/hot initialization, communicator reuse, AVG semantics and version-specific defaults.
Run these files through torch.distributed.run with eight processes and --experimental
in the supported TE/CUDA environment. Repository pre-commit and syntax checks apply
at this reconstructed commit. Historical source-branch regressions passed, but are not
represented as newly rerun tests. New single-node proxy FSDP/DDP runs validate the final
assembled stack; a run with UB disabled does not independently exercise UB registration.
