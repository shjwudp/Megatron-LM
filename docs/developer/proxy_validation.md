# Reconstructed proxy branch validation

The reconstructed runtime at `708e44495` completed two single-node, eight-GPU proxy runs.
Subsequent calculator/documentation changes do not alter this runtime. FSDP uses the
MFSDPv2 hybrid path with dense ZeRO-2 and expert ZeRO-1; DDP uses distributed optimizer.
Both use VPP, combined 1F1B, EP overlap, deferred wgrad, the nine recompute selections and
a standalone MTP virtual chunk. Manual GC remains at interval 30.

## Local dump acceptance

Both runs explicitly set ENABLE_PP_TIMER_DUMP_ONLY=1, ENABLE_PP_TIMER=0,
ENABLE_PP_TIMER_INTERVAL=20, ENABLE_PP_TIMER_ITER=99-99 and ENABLE_PP_TIMER_CPU=1.
Every rank has all zero-based Timer iterations 0–39, including steps outside that sampling
window. Each GPU file identifies dump_only=true; CPU files and complete start/end pairs
are present. EventTracker contains no uploader or metrics-service dependency.

| Path | Steps / exit | GPU + CPU Timer files | Valid intervals / invalid | CUDA kernels / complete NVTX | Max allocated / reserved / sampled device GiB |
| --- | --- | --- | --- | --- | --- |
| DDP | 40 / 0 | 320 + 320 | 129808 / 0 | 118698 / 36236 | 71.572 / 76.354 / 87.452 |
| HFSDP | 40 / 0 | 320 + 320 | 176960 / 0 | 121933 / 49423 | 81.949 / 109.926 / 122.710 |

Both runs completed with finite loss/gradient norms, no NaN or skipped iteration, and four
readable allocator snapshots each. Forty mock batches have identical valid/physical token
counts, padding and document-length workload across modes. Current allocated varies by
at most 34.098MiB per rank over the run; the GC30 boundary changes it by zero bytes.
Lifetime allocator maxima and sampled device peaks have distinct definitions.

## Captured recompute and FSDP communication NVTX

The HFSDP trace contains these complete ranges, summed across all ranks in the two-iteration
capture window (human iterations 5–6). These are measured labels, not only code coverage.

| NVTX label | Count |
| --- | --- |
| recompute csa_core_output | 128 |
| recompute csa_o_proj | 128 |
| recompute mla_up_proj | 128 |
| recompute dense_fc1_act | 32 |
| recompute moe_fc1 / moe_act / shared_experts | 96 each |
| recompute layernorm | 224 |
| optimizer mfsdp param gather axis0 / axis1 | 344 / 248 |
| optimizer mfsdp grads reduce scatter axis0 / axis1 | 152 / 512 |
| ncclAllGather / ncclReduceScatter | 620 / 408 |

Recompute labels additionally identify layer, microbatch and virtual stage when available.
NVTX spans describe CPU submission scopes; inspect correlated CUDA/NCCL kernels for device
execution and overlap. Timer event intervals can include waits on the measured stream.

## Unit and browser checks

- Dense-only chunk validation: 2 passed.
- GPT builder and schedule boundary/lifetime suite: 78 passed.
- Local selective recompute: 12 passed; CP2 O/LSE replay: 4 passed per rank.
- Round-robin count and straight-through gradients: 2 passed.
- Event Tracker and CSA FLOPs: 77 passed, including dump-only on/off ordinary enable states.
- Calculator: 19 Node regressions passed. Browser checks passed for two presets, exact
  default batch, EP2 switching, ZeRO selection, JSON round-trip, auto batch and old-preset fallback.

The initial CP2 test launch used one process and failed topology validation; its two-process
rerun passed. Initial browser tooling needed its Chromium runtime and the JSON panel had
to be opened before interaction; these were corrected before the successful browser run.
The 5% calculator snapshot calibration uses historical proxy measurements, not a new
16-node test. Raw experiment identities, private paths and launchers are outside Git.

## Limits and provenance

The training harness is frozen outside the feature history, as requested. No production
model preset or original model comparison is included. The five upstream PRs remain
independent, retaining authors and source revisions. Their source numerical/UB results
are historical; this new no-UB training pair does not revalidate UB registration or
checkpoint round-trip. No new 16-node task was submitted and no upstream CI was run.
Every feature commit ran repository pre-commit. Model/optimizer/schedule runtime sources
match the input branch; the only functional runtime delta is local dump-only control.

Machine-readable results are in [proxy_validation.json](proxy_validation.json).

The calculator/documentation commit is outside the Python hook filters; Node, browser,
JSON and local-link checks cover those files. Source worktree hashes were preserved.
