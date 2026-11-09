# Result

Validated on 2026-11-07. MFSDP now receives the configured Muon algorithm and shares
logical split math with TP=1 DDP. Both paths preserve the reference language
parameter policy. The reference also uses a different attention implementation;
CSA-specific parameters follow the migration rules documented in
[the developer guide](../../developer/proxy_muon_optimizer.md).

## Executed checks

All GPU tests ran in the comparison image with the pinned TE and EO 0.3 overlays
on one node with eight GPUs.

- 29 focused cases per rank: existing FSDP Muon regression cases, independent
  polynomial checks, metadata copying, the actual MCore factory and ordinary/hybrid
  sharding. The two distributed alignment cases initially failed because the test
  reconstructed packed RowAtomic parameters with ordinary DTensor rules. The
  corrected DBuffer reconstruction passed on all ranks. Five-step losses and final
  complete weights match the unsharded reference exactly.
- 13 existing emerging-optimizer cases per rank: constructor/registry dispatch,
  adaptive-Muon kwargs, coefficient types and QKV splitting. All passed.
- Independent legacy comparison: original reference Muon and EO v0.1.0 definitions
  were loaded separately from the new implementation. Whole matrices, FC1, EH,
  indexer heads and fused-QKV heads were checked with normal, tiny and zero
  gradients over three optimizer steps. Momentum, NS input/output, decayed weights
  and final updates were bit-exact under the same installed PyTorch/CUDA stack.
  All seven deliberately incorrect settings were detected by negative controls.
- Local HFSDP and DDP proxy training: each completed six iterations on eight GPUs,
  exit code zero, finite loss, no skipped or NaN iterations. VPP, packed sequence,
  EP overlap, deferred weight gradients and selective recomputation were enabled.
  Each rank's runtime optimizer audit passed. Logical Adam/Muon routing and split
  metadata match across the two backends. Both runs used identical packed-data
  profiles, seeds and generated layouts.
- Each local run produced 48 CPU and 48 GPU Event Tracker files, per-rank memory
  metrics for all six iterations, and four representative snapshot files.
- Repository pre-commit passed Black, Pylint and isort; Python syntax and
  `git diff --check` passed. Training CLI files are outside the hook's formatter
  filter and were checked separately for syntax.

The local training check establishes functionality, not steady-state throughput
or convergence equivalence. The independent comparison does not reproduce the
reference container's older PyTorch/CUDA runtime. Checkpoint conversion was not tested.

## Reproduction

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q --experimental \
  tests/unit_tests/test_muon_matrix.py \
  tests/unit_tests/distributed/mfsdp_v2/test_muon.py \
  tests/unit_tests/distributed/mfsdp_v2/test_mcore_dense.py::test_muon_factory_forwards_effective_config

python -m torch.distributed.run --standalone --nproc_per_node=8 -m pytest -q \
  tests/unit_tests/test_emerging_optimizers.py \
  -k 'shared_muon_kwargs or adaptive_muon_kwargs or muon_entry or test_muon_optimizer_qkv_split or test_muon_optimizer_coefficient_types'
```

The external experiment directory `proxy-optimizer-alignment-20261107` contains
`reference-audit.md`, `reference-parity.py`, source hashes, per-rank optimizer audits,
local training logs and `local-validation-results.json`. Workload dimensions and
submission scripts remain outside this feature repository.

## Completed comparison and remaining limits

The final HFSDP and DDP comparisons both completed all training iterations.
Their optimizer parameter routing and effective configuration passed audits on
all participating ranks. Stable optimizer means were 2829.360/2536.151 ms;
forward/backward means were 16978.133/14448.800 ms. The previous DDP optimizer
disadvantage disappeared, while HFSDP forward/backward remained slower.

The separately shared experiment report contains the workload configuration,
throughput and memory statistics, Event Tracker distributions, overlap accounting
and Nsight Systems evidence. These measurements characterize that workload and
hardware; they do not establish a general performance ordering.

No upstream CI, checkpoint conversion or additional training was run when
preparing the private sharing branch. Its production source and unit tests were
checked against the immutable archive from the completed comparisons; see the
[release provenance](../../developer/proxy_experiment_release.md).
