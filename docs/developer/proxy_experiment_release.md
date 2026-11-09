# Proxy training experiment source

This branch contains the Megatron source used by the completed HFSDP and DDP
comparisons, including the Adam/Muon alignment fix. The experiment's model
configuration, launch harness and profiling artifacts are distributed separately.

## Source provenance

- Upstream: <https://github.com/NVIDIA/Megatron-LM>.
- Upstream ancestor: `a0ef64615cc3d2574ffbaac70f324ee365f5ed34`.
- Feature baseline: `3568bd3eca5b6bc5f14e25ab7314395974fa813b`.
- Executed source archive SHA-256:
  `b792bdb5942234eb7321bb4ceacd9e8cccc0ac53460a84b25566c9dcef232604`.
- Executed dependency overlay SHA-256:
  `2dd48818a951841d033db3f6c8183f8f8883958c7f0e06c6df0fec772d207b33`.

Both runs used the same archive. It was frozen before the optimizer fix was
committed. The sharing branch restores the fix from that archive. All 2,625
regular files under `megatron/`, `tests/` and `gpt_builders.py`, plus two symlinks,
were compared with the archive and matched. The documentation has been updated
to record the completed validation and to remove local infrastructure references.
The archive hash identifies the original full experiment bundle, including its
external launch harness; it is not the hash of a checkout or a GitHub archive.

The branch retains the existing feature history. Upstream PRs #8684, #8782,
#8856, #8918 and #8950 remain separate commits. See the
[feature index](proxy_model_changes.md) and
[optimizer validation](../tasks/proxy-optimizer-alignment/Result.md).

## Reproduction inputs

The experiment artifact package must accompany this source for reproduction:

- Common workload configuration and the mode-specific effective arguments.
- Packed mock-data generator, sampled document-length profile and fixed seeds.
- Launch/training harness and optimizer-audit script.
- Runtime versions and the pinned dependency overlay.
- Per-rank audits, logs, memory metrics and profiling artifacts.

The optional kernel cache only accelerates startup on a compatible runtime.
Site-specific paths and cluster launch adapters need adjustment on another
installation. The code does not embed cluster credentials or job submission
records. Licenses and upstream attribution remain in their original files.

## Validation scope

The optimizer task result records the executed eight-GPU numerical and training
checks and the completed larger comparisons. Preparing this source release does
not rerun training. Source equality, repository pre-commit hooks and whitespace
checks are performed on the sharing branch before publication.
