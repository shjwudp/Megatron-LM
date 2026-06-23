# Activation Checkpointing × FSDP CUDA Graph Runner — Design

> **Status:** superseded. The `MFSDP_CG_NO_GRAD_FWD` approach
> (see `cuda_graph_runner.py` docstring for `_CG_NO_GRAD_FWD`) is a
> strictly better solution: it achieves the same forward-graph
> savings as `torch.utils.checkpoint` without the bwd-graph growth
> or extra recompute, because the bwd-capture path already re-runs
> the forward with grad enabled to build its own autograd tape.
>
> This document is kept for historical context and to inform future
> user-checkpoint-detection work (Case B below).

## Problem statement

The full-snapshot analysis (see `cuda_graph_memory_analysis.md`) shows
**inductor activation intermediates pinned in the graph pool** are the
single largest CG overhead bucket — +18.7 GB across 60 transformer
layers (312 MB/layer). These are SavedVariables that the captured
forward accumulates for backward. Because `_capture_backward_and_run`
already re-runs the forward to build its own autograd tape, the
forward-graph SavedVariables are dead weight at replay time.

## Why no_grad capture supersedes torch.utils.checkpoint

The key realization: the bwd-capture path ALREADY recomputes the
forward (see `_capture_backward_and_run` line ~913). The forward-graph
SavedVariables are never read by the captured backward.

| Approach | Fwd graph size | Bwd graph size | Recomputes in bwd capture |
|---|---|---|---|
| Legacy (grad-enabled fwd capture) | 312 MB/layer (dead SavedVars) | recompute + bwd | 1 (runner's) |
| `torch.utils.checkpoint` wrap | small (Holder placeholders) | recompute + recompute + bwd | **2** (runner's + checkpoint's unpack) |
| **`no_grad` fwd capture** | **~0** (no SavedVars) | recompute + bwd | **1** (runner's, unchanged) |

`no_grad` capture is strictly better than checkpoint here:
- same fwd-graph savings as checkpoint
- no extra recompute in bwd capture
- no bwd-graph growth
- bonus: capture-time behavior matches replay-time behavior (PyTorch
  runs `torch.autograd.Function.forward` with grad disabled, so
  `fwd_graph.replay()` already executes under no_grad at runtime)

**IMPLEMENTED**: `MFSDP_CG_NO_GRAD_FWD=1` (default ON). See
`cuda_graph_runner.py` env-var docstring for details.

The sections below are preserved for the user-side checkpoint
detection discussion (Case B), which is orthogonal to no_grad_fwd.

Users may want to enable activation checkpointing in TWO ways:

- **Case A — runner-side wrapping:** User sets
  `MFSDP_CG_USE_CHECKPOINT=1`. The CG runner itself wraps
  `module.forward` with `checkpoint(..., use_reentrant=False)` during
  `capture_forward` (already implemented).
- **Case B — user-side wrapping:** User wraps `module.forward`
  themselves before passing the module to the FSDP CG runner (common
  pattern when integrating with HF diffusers / TransformerEngine's
  `fp8_checkpoint` / PEFT-style wrappers).

Both cases need to compose correctly with the runner's separate
fwd/bwd CUDA graph capture. Case B raises a detection question: *can
the runner detect that the user already wrapped with checkpoint, and
should it?*

## Detection options (empirical assessment, PyTorch 2.4)

| Strategy | Reentrant | Non-reentrant | Notes |
|---|---|---|---|
| Walk output `grad_fn` chain for `CheckpointFunctionBackward` | ✅ 100% reliable | ❌ invisible | Reentrant uses a custom autograd Function whose backward node IS visible in the graph. Non-reentrant uses `_NoopSaveInputs` only as a sentinel — its result isn't reachable from output's `grad_fn`. |
| Probe with `saved_tensors_hooks` interposition | ✅ visible via reentrant's `_recomputation_hook` | ❌ doesn't work | `saved_tensors_hooks` does NOT nest: the most-recently-registered pair replaces earlier ones. Our outer hook is inactive while checkpoint's inner hook runs. |
| Inspect `SavedTensor._data` for `_Holder` instances | ❓ untested | probable | Requires private C++ API access (`torch._C._autograd.SavedTensor`); kept opaque by the autograd engine. |
| `inspect.getsource(module.forward)` for `'checkpoint'` | heuristic | heuristic | Works for direct user code, fails for closures defined in REPL/Jupyter (no source file), and misses checkpoint nested inside third-party modules (TE, HF). |
| `torch.autograd._is_checkpoint_valid()` | ❌ no | ❌ no | Only returns state during the backward recompute phase, not at capture time. |

**Conclusion:** there is no robust, portable way to auto-detect
non-reentrant checkpoint (the PyTorch-recommended default) from
outside the user's forward. Source inspection is the only heuristic
available, and it covers ~70% of real usage patterns.

## Proposed design

### 1. Explicit flag-based opt-in (recommended, no auto-detect)

Don't try to auto-detect. Expose two **mutually exclusive** env vars:

```
MFSDP_CG_USE_CHECKPOINT=1
    Runner wraps module.forward with checkpoint(use_reentrant=False)
    during capture_forward. Composes with MFSDP_CG_COMPILE_FWD=1 as
    torch.compile(checkpoint(orig)). Restored to original forward
    before install() patches the user-facing forward.
    IMPLEMENTED.

MFSDP_CG_USER_CHECKPOINT=1
    Declares that the user ALREADY wraps module.forward with
    checkpoint (any variant). Runner skips its own checkpoint wrap,
    adjusts backward capture (see §3 below), and runs best-effort
    detection (see §2 below) to warn if the declared pattern doesn't
    match what was actually found.
    NOT YET IMPLEMENTED.
```

If both flags are set, raise at startup — they're mutually exclusive.

If neither flag is set, runners assumes no checkpoint and behaves as
today (eager or `torch.compile` per `MFSDP_CG_COMPILE_FWD`).

### 2. Best-effort detection (for warnings, not enforcement)

When `MFSDP_CG_USER_CHECKPOINT=1` is set, runner runs a detection
probe in `capture_forward` (before popping hooks) to cross-check:

```python
def _detect_checkpoint(m: torch.nn.Module, sample_args) -> Optional[str]:
    """Returns 'reentrant', 'non_reentrant', or None.

    Heuristic; not authoritative for non-reentrant (may return None
    even when checkpoint is active).
    """
    # 1. Reentrant: probe forward + grad_fn walk.
    probe_inputs = tuple(
        torch.randn_like(t).requires_grad_(t.requires_grad) if torch.is_tensor(t) else t
        for t in sample_args
    )
    with torch.no_grad(enabled=False):
        out = m(*probe_inputs)
    found = _walk_grad_fn_for(out.grad_fn, lambda n: type(n).__name__ == 'CheckpointFunctionBackward')
    if found:
        return 'reentrant'

    # 2. Non-reentrant: source inspection of m.forward (best-effort).
    try:
        src = inspect.getsource(m.forward)
        if 'use_reentrant=False' in src or 'use_reentrant= False' in src:
            return 'non_reentrant'
        if 'checkpoint(' in src or 'torch.utils.checkpoint' in src:
            return 'unknown_variant'
    except (OSError, TypeError):
        pass

    # 3. Optional: probe SavedTensor._data for _Holder (private API,
    #    may break across torch versions — gate behind try/except).
    try:
        from torch.utils.checkpoint import _Holder
        # ... SavedTensor introspection logic ...
    except Exception:
        pass

    return None
```

When `MFSDP_CG_USER_CHECKPOINT=1` but detection returns `None`,
emit a `logger.warning` (not an error — non-reentrant is undetectable
in many cases).

When detection returns `'reentrant'`, raise — reentrant checkpoint is
incompatible with CG (see §3).

### 3. Compatibility rules per checkpoint variant

Reentrant (`use_reentrant=True`) — **INCOMPATIBLE with CG**.

Reentrant checkpoint's backward re-runs forward via
`torch.autograd.grad(...)` with a fresh autograd tape. This re-run
runs OUTSIDE the captured `bwd_graph` (it happens inside
`CheckpointFunction.backward`, which the engine calls), so the
captured backward graph doesn't include the recompute kernels —
but it does reference the recomputed intermediates' addresses, which
aren't in any pool the graph can see. The result is silently wrong
gradients or replay crashes. Raise an error if detected.

Non-reentrant (`use_reentrant=False`, the default) — **COMPATIBLE**.

Non-reentrant checkpoint uses `saved_tensors_hooks` whose unpack hook
runs the recomputation lazily, inside the engine's normal backward
dispatch. The flow under CG:

| Phase | What happens | Captured into |
|---|---|---|
| Fwd capture (runner's `capture_forward`) | `checkpoint(orig_fwd, x)` runs `orig_fwd` once. Inductor intermediates allocated + used, then freed at the checkpoint boundary. Only `_Holder` placeholders saved on SavedTensor nodes. | `fwd_graph` contains the kernel launches but NOT the per-op intermediates (they were freed at boundary). Net fwd graph memory drops by ~312 MB/layer. |
| Bwd capture (`_capture_backward_and_run`) | Runner re-runs forward via `_call_module` to build a fresh autograd tape — this re-runs `checkpoint(orig_fwd, x)` which re-runs `orig_fwd` AGAIN. Recompute intermediates allocated inside the `bwd_graph` capture context. Then `torch.autograd.grad(...)` runs the actual backward, which triggers checkpoint's unpack hook → recompute → another forward pass. | `bwd_graph` now contains: (i) the recompute kernels (from the runner's pre-capture forward) AND (ii) the recompute kernels (from checkpoint's unpack). These likely double-allocate intermediates. |

**Net memory effect (Case A, runner-side wrap):** fwd graph shrinks by
~312 MB/layer (good), bwd graph grows by roughly the same amount (bad,
because the recompute happens inside bwd capture). The two graphs
share the same pool, so freed fwd addresses MIGHT be reused by bwd —
but the recompute timings differ, so reuse isn't guaranteed. The net
is empirically uncertain; needs GPU measurement.

**Mitigation for the bwd-graph growth:** when
`MFSDP_CG_USER_CHECKPOINT=1` (or `MFSDP_CG_USE_CHECKPOINT=1`), skip
the runner's forward re-run in `_capture_backward_and_run` and instead
let the captured `bwd_graph` rely on checkpoint's own recompute. This
is a deeper change — see §4.

### 4. Open design questions

**Q1: Should `_capture_backward_and_run` skip its forward re-run when
checkpoint is active?**

Today: the forward re-run is needed to build a fresh autograd tape
(since `fwd_graph.replay()` doesn't build a tape). With checkpoint,
the re-run ALSO triggers checkpoint, doubling the compute. Options:

- **Skip the re-run entirely** — let checkpoint's unpack hook do the
  only recompute. Requires restructuring `_capture_backward_and_run`
  to capture backward starting from the live fwd output (via the
  installed `_patched_fwd`'s autograd graph) rather than a fresh
  re-run. Risk: the autograd graph from `_patched_fwd` lives outside
  the capture stream and may not be address-stable.
- **Re-run with checkpoint disabled** — temporarily patch
  `module.forward` back to `orig_fwd` (the inner un-checkpointed
  body) during the bwd-capture re-run, so the recompute happens only
  once (during checkpoint's unpack). Cleaner but brittle.

**Q2: Should we recommend Case A (runner-side wrap) or Case B
(user-side wrap) to users?**

- Case A is simpler for users (one env var) but couples checkpoint
  policy to the FSDP runner — undesirable if the user wants selective
  checkpointing (e.g., only every other layer).
- Case B is more flexible but requires the user to understand the
  interaction; the runner can't fully auto-detect their wrapping.

Recommend: **Case B as the primary path**; Case A as a convenience
shortcut. Document Case B clearly with the constraint
`use_reentrant=False` mandatory.

**Q3: How does this interact with `MFSDP_CG_COMPILE_FWD=1`?**

Composition `torch.compile(checkpoint(orig))` is the PyTorch 2.x
recommended ordering — compile sees the whole checkpoint region and
fuses the recompute path. For Case B (user-side wrap), the user
already has `checkpoint(orig)` installed; the runner can still apply
`torch.compile` around it (the runner-side compile layer wraps
whatever forward is installed). For Case A, the runner controls both
layers and applies them in the correct order.

## Summary table

| User does | Runner flag | Detection | Behavior |
|---|---|---|---|
| No checkpoint, no compile | (none) | n/a | legacy: capture eager forward |
| No checkpoint | `MFSDP_CG_COMPILE_FWD=1` | n/a | capture `torch.compile(orig)` |
| No checkpoint | `MFSDP_CG_USE_CHECKPOINT=1` | n/a | runner wraps, captures `checkpoint(orig)` |
| No checkpoint | both flags | n/a | runner wraps, captures `torch.compile(checkpoint(orig))` |
| User wraps with `checkpoint(use_reentrant=False)` | (none) | warn (heuristic) | capture user's wrapped forward as-is; may have bwd-graph growth issue |
| User wraps with `checkpoint(use_reentrant=False)` | `MFSDP_CG_USER_CHECKPOINT=1` | verify + warn if undetected | same as above, but runner adjusts bwd-capture strategy (Q1) |
| User wraps with `checkpoint(use_reentrant=True)` | any | raise | incompatible — refuse to capture |
| User wraps with checkpoint AND `MFSDP_CG_USE_CHECKPOINT=1` | (both) | error at startup | mutually exclusive — refuse |

## Recommended next steps

1. Add `MFSDP_CG_USER_CHECKPOINT=1` flag (mutually exclusive with
   `MFSDP_CG_USE_CHECKPOINT=1`).
2. Implement `_detect_checkpoint(m, sample_args)` (§2) for warnings.
3. Refuse reentrant checkpoint with a clear error.
4. Measure Case A (`MFSDP_CG_USE_CHECKPOINT=1`) full-snapshot to
   confirm the fwd-graph savings and bwd-graph growth hypothesis.
5. Based on measurement, decide Q1 (skip forward re-run in bwd
   capture).
6. If Case B is the recommended path, add a public API on
   `FSDPCudaGraphRunner` (e.g.
   `runner.declare_user_checkpoint(variant='non_reentrant')`) so
   callers don't have to use env vars.

## Implementation locations

- `cuda_graph_runner.py`:
  - `_CG_USE_CHECKPOINT` env var (already implemented, line 154).
  - `_captured_fwd_was_compiled` / `_orig_fwd_body` attrs (already
    implemented, `__init__`).
  - Forward wrapping in `capture_forward` (already implemented, §3b
    block at line ~508).
  - **TODO:** `_CG_USER_CHECKPOINT` env var.
  - **TODO:** `_detect_checkpoint()` function.
  - **TODO:** reentrant detection + raise in `capture_forward`.
  - **TODO:** conditional bwd-capture strategy.
- `cuda_graph_memory_analysis.md`: updates after §5 measurement.
