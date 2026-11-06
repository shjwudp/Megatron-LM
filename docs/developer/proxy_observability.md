# Proxy model observability and analysis

This single guide consolidates Event Tracker, PP Timer, compact NVTX, recompute ranges,
CSA FLOPs, per-iteration memory reporting and MFSDPv2 diagnostic/snapshot tooling.
Routing controls have their own feature because forcing expert balance changes training.

## Local PP Timer output

This branch's EventTracker has no metrics-server uploader. It writes JSON at the end of
a completed train_step. The separate internal PP Timer implementation is not imported.

```bash
export ENABLE_PP_TIMER_DUMP_ONLY=1
export ENABLE_PP_TIMER_DUMP_DIR=/tmp/proxy-timer
export ENABLE_PP_TIMER_CPU=1
```

DUMP_ONLY enables the timer on every completed iteration, including when ENABLE_PP_TIMER
is unset or zero. It overrides ENABLE_PP_TIMER_ITER/INTERVAL sampling. No upload path is
called. Without DUMP_ONLY, ENABLE_PP_TIMER=1 retains the existing sampled behavior:
ENABLE_PP_TIMER_ITER selects a zero-based inclusive range; otherwise INTERVAL defaults
to 20. Setting DUMP_ONLY=0 restores that behavior. Dump-only mode itself adds no new
per-module synchronization; collection synchronizes CUDA once at the iteration boundary.
Every-iteration collection therefore has measurement overhead and is not a throughput baseline.

Output files are event_times_iter<I>_rank_<R>.json and, with CPU timing enabled,
event_times_cpu_iter<I>_rank_<R>.json. I is zero-based; human training logs are one-based.
GPU files include reference_unix_ms, events, synchronize_ms, iteration, rank, cpu_enabled
and dump_only. Writes complete before train_step returns. An incomplete/failed step does
not manufacture successful endpoints. CUDA graphs remain unsupported by this eager tracker.

## NVTX and interpretation

Native profiling windows emit iteration, paired schedule, layer, short node and selective
recompute scopes even on steps without Timer samples. NVTX-only tracing creates no CUDA
timing events. NVTX_EVENT_TRACKER_VERBOSE=1 restores detailed duplicate Timer envelopes;
the default compact hierarchy removes those envelopes while preserving Timer keys.
Recompute labels retain strings rather than tensors or model objects.

Attention, dense/routed/shared MLP, dispatch/combine and delayed wgrad have forward/backward
ranges. HFSDPv2 annotates parameter gather and gradient reduction by mesh axis; Muon owner
transfers are also observed. DDP asynchronous collectives use a private observer stream
and preserve the original Work handle. Overlapping durations cannot be added to obtain
iteration time: GPU intervals include stream stalls, and CPU intervals measure submission.

## FLOPs and memory metrics

CSA model FLOPs account for Q/KV/grouped-O projections, compressed/sliding pairs, indexer
forward/wgrad, sparse loss and MTP. Packed document lengths use a length-weighted pair
approximation; equal-length cases are exact. This is model FLOPs, not all executed kernel
work or hardware MFU. Recompute, optimizer, communication and elementwise work are excluded.

The training logger supports elapsed iteration time and TFLOPS/GPU. Memory reports must
separate current allocated/reserved, lifetime maxima and sampled whole-device usage.
Device-minus-reserved must use simultaneous samples; subtracting unrelated maxima is invalid.
The summary tool validates complete rank/window coverage before comparing proxy results.

## Tools and buffer interpretation

- tools/profiling/summarize_pp_timer.py validates endpoints and emits a CPU/GPU timeline;
  plot_pp_timer.py renders selected intervals. They accept the same local dump files.
- tools/profiling/summarize.py summarizes training throughput and allocator/device JSONL;
  prepare_snapshots.py makes traceable analysis copies without changing raw snapshots.
- tools/mfsdp_v2/dump_optimizer_groups.py inspects matrix/scalar optimizer assignment;
  fsdp_buffer_report.py reports dtype, mesh, placements and local bytes;
  dump_grad_dtypes.py observes autograd, staging and accumulated gradient precision.
- fsdp_snapshot_trace.py and fsdp_rs_timeline.py replay allocator lifetimes by callsite;
  csa_snapshot_audit.py checks CSA saved-state/recompute attribution.

MXFP8 parameter gather stores the supported quantized compute representation. Weight
quantization workspaces are distinct from persistent parameters. FP32 main-grad settings
control staging/reduction/accumulation, while autograd may initially produce BF16 wgrad.
A staging buffer can coexist with an asynchronous predecessor; count full unit storage
before sharding. Expert-DP size one can still pack gradients. See the calculator's own
consolidated guide for the bounded temporary-memory estimate and its limitations.

## Verification

Run test_event_tracker.py and training/test_csa_flops.py via torchrun. The timer suite
covers disabled/sampled/NVTX-only operation, consecutive capture steps, balanced scopes,
backward and recompute ranges, async completion, and DUMP_ONLY with both ordinary enable
states. Dump tests verify files exist immediately after each returned step despite a
conflicting sampling window. FLOPs tests compare against explicitly enumerated causal pairs.
Single-node FSDP and DDP verification additionally checks every rank/iteration, finite
training, complete Timer endpoints and actual nsys CUDA/NVTX records. Historical reports
are not counted as reconstructed-branch validation.

Reconstructed branch result: 77 observability/FLOPs cases passed, including the new local-dump mode.
