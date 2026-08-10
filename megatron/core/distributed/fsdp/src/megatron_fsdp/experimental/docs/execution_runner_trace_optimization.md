# FsdpExecutionRunner: Trace Path and Optimization Path

**Status:** Design proposal for the M-FSDP v2 execution-order runner.

**Audience:** Distributed-training developers working on Megatron-FSDP v2 with
pipeline (PP/VPP) and expert-parallel combined-1F1B schedules.

## 1. Problem

Under the combined-1F1B + VPP schedule, parameter consumption is
occurrence-based rather than a single traversal of the module tree:

- The same FSDP unit can be consumed in forward and backward (e.g. `F L56`
  and `B L56` for different microbatches).
- Model chunks interleave, and warmup/steady/cooldown differ per pipeline
  rank.
- The schedule fires one fine-grained hook per sub-module (dense layer,
  experts), so the same `FsdpModule` can be touched several times per pass.

The static `forward_order` / `backward_order` sequences cannot express this
runtime path, so M-FSDP v2 uses a per-context `FsdpExecutionRunner` that
**traces** the real execution and **replays** it to drive prefetch. This
document defines two cooperating paths inside the runner:

1. **Trace path** — records the real op stream (consume and reshard events).
2. **Optimization path** — during replay, translates the real ops into an
   optimized plan (e.g. skip a reshard + all-gather pair when the traced
   schedule re-consumes the same module immediately).

## 2. Design: two paths in one runner

```text
                 FsdpContext (one per rank, shared across VPP chunks)
                              |
                     FsdpExecutionRunner
                    /                    \
            Trace path                 Optimization path
   records real ops during      replay validates against the
   the first global batch       trace and returns directives
        (consume, reshard)      (prefetch target, skip reshard)
                              |
                    FsdpModule entry points
              pre_forward / pre_backward / unshard_parameters
              _reshard_parameter_groups
```

### 2.1 Trace path (global batch 1)

The runner records every fine-grained execution event as a `RunnerEvent`:

```python
class EventKind(Enum):
    CONSUME = auto()   # module params are consumed by compute
    RESHARD = auto()   # module params are released after compute

@dataclasses.dataclass(frozen=True)
class RunnerEvent:
    kind: EventKind
    module: FsdpModule
    orientation: str | None   # rowwise/colwise; None for reshard
```

The trace is the ordered list of events observed during the first global
batch:

```text
[CONSUME(L2, rowwise), RESHARD(L2), CONSUME(L2, colwise), RESHARD(L2),
 CONSUME(L0, rowwise), RESHARD(L0), CONSUME(L0, colwise), RESHARD(L0), ...]
```

During tracing no prefetch is issued (demand-only) and no reshard is
optimized away. The training loop calls `complete_trace()` at every
global-batch boundary (via the optimizer step); the first non-empty trace
compiles into the replay cycle.

### 2.2 Optimization path (global batch 2+)

During replay, each real op is validated against the traced event at the
current position (`_replay_index`), and the runner returns an optimization
directive:

- `consume_and_get_next(module, orientation) -> (module, orientation) | None`
  — validates the consume, advances the cursor, and returns the next
  **consume** event (skipping intervening reshard events) as the prefetch
  target.
- `reshard(module) -> bool` — validates the reshard, advances the cursor,
  and returns whether the actual reshard can be **skipped** so the storage
  stays resident.

A mismatch (wrong event kind, module, or orientation) is a divergence:
the runner clears the trace, re-traces from that event, and degrades to
demand-only execution until a full cycle matches again.

## 3. Optimization: skip reshard + unshard on immediate reuse

### 3.1 Rule

During replay, when a reshard for module `M` is about to execute, the runner
looks at the traced event that follows the reshard:

```text
... RESHARD(M)  CONSUME(M, orient) ...
```

If the next traced consume is the **same module with the same orientation**,
the storage is re-consumed immediately, so the reshard is unnecessary. The
runner returns `True` from `reshard(M)` and the module keeps its unsharded
storage resident. The following consume then finds the storage already
materialized and skips the all-gather.

### 3.2 Why same orientation?

M-FSDP v2 MXFP8 parameter groups keep separate row-wise (forward GEMM) and
column-wise (backward GEMM) payloads. Keeping storage resident across an
orientation change would leave the wrong payload materialized, so the
optimization only applies when the immediate re-consume uses the same
orientation.

### 3.3 When not applied

- Different orientation on the immediate re-consume.
- Another module's consume intervenes between the reshard and the re-consume.
- Default mode (`use_trace_replay=False`): the runner stays idle and every
  reshard is executed normally.
- Tracing phase or after a divergence.

### 3.4 Example

Traced cycle (forward-only pass over two layers):

```text
[CONSUME(L0,row), RESHARD(L0), CONSUME(L0,row), RESHARD(L0),
 CONSUME(L1,row), RESHARD(L1), CONSUME(L1,row), RESHARD(L1)]
```

Replay:

| Real op | Runner directive |
|---|---|
| `consume(L0,row)` | prefetch `(L0,row)` (next consume) |
| `reshard(L0)` | **skip** — next consume is `(L0,row)` |
| `consume(L0,row)` | storage resident, no all-gather |
| `consume(L1,row)` | prefetch `(L1,row)` |
| `reshard(L1)` | **skip** |
| `consume(L1,row)` | storage resident, no all-gather |

Saves one all-gather and one reshard per module per pass.

## 4. Interface

Public API of `FsdpExecutionRunner` (owned by `FsdpContext`):

| Method | Path | Purpose |
|---|---|---|
| `consume_and_get_next(module, orientation)` | both | consume directive + prefetch target |
| `reshard(module) -> bool` | optimization | skip-reshard directive |
| `reset_pass(module)` | trace | re-enable a fresh consume after reshard |
| `complete_trace()` | trace | compile the cycle at the batch boundary |
| `report()` | diagnostics | replay statistics |
| `phase`, `is_tracing`, `use_trace_replay` | — | runner state |

`FsdpModule` integration:

```python
# unshard_parameters (consume entry point)
prefetch = self.context.runner.consume_and_get_next(self, orientation)
if prefetch is not None:
    next_module, next_orientation = prefetch
    next_module._unshard_parameter_groups(next_orientation)

# _reshard_parameter_groups (release entry point)
self.context.runner.reset_pass(self)
if self.context.runner.reshard(self):
    return  # storage stays resident
for group in self._parameter_groups:
    group.reshard_parameters()
... # release storage on the all-gather stream
```

## 5. Correctness arguments

- **Consume validation** ensures the real schedule still matches the traced
  cycle; divergence falls back to demand-only, never skipping a collective.
- **Reshard skip** is safe only for an immediate same-module,
  same-orientation re-consume, so the materialized payload is always the one
  the next compute reads.
- **Dedup** (`_consumed_this_pass`) keeps the trace at one consume per module
  per pass despite per-sub-module hooks; `reset_pass` is called on reshard so
  the next pass records a fresh consume.
- **Memory** is bounded: skipping a reshard keeps at most one extra module's
  storage resident, and only while it is immediately reused.

## 6. Open questions

1. Should the reshard-skip policy be extended to a *window* (keep resident if
   re-consumed within N events) instead of strictly immediate? This trades
   memory for fewer all-gathers and needs a residency budget.
2. Should the optimization path also skip the all-gather for a module that is
   resident but whose reshard was *not* skipped (e.g. prefetched modules)?
3. How should the optimization path interact with the MXFP8 scale-inverse
   grids when a payload is kept resident across optimizer steps?
4. Should `complete_trace()` compile a more elaborate plan (keep/resident
   windows, prefetch distances) instead of the event cursor? See
   `vpp_1f1b_design.md` §3.

## 7. Sources

- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/execution_runner.py`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/docs/vpp_1f1b_design.md`
- `megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/docs/mfsdp_v2_vpp2_1f1b_schedule.md`
