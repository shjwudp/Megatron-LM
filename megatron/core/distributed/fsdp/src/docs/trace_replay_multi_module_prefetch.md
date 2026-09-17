# Trace-and-replay: multi-module all-gather prefetch

Status: **design proposal, draft for review** — no implementation in this PR.

Base: `fix/tr161-vpp-trace-replay` @ `757668734` (the merge of the trace-and-replay
scheduler and its orientation fix).

Related: the trace-and-replay scheduler, its `_build_plan` reshard-skip logic, and
the orientation work that made per-phase unshard narrowing take effect.

---

## 1. Problem

On the trace-and-replay path the all-gather prefetch depth is hard-coded to **one
module**:

```python
# megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/schedule.py
# PlanOp
prefetch_after: tuple[FsdpModule, str | None] | None = None

# _build_plan, step 3: after each WAIT_UNSHARD, prefetch exactly the next distinct
# module that will be unsharded
for j in range(i + 1, n):
    if (events[j].kind is OpKind.ISSUE_UNSHARD
            and events[j].module is not events[i].module):
        plan[i].prefetch_after = (events[j].module, plan[j].orientation)
        break
```

So during module *k*'s compute exactly one gather (for *k+1*) is in flight. If
`compute(k) < gather(k+1)`, the wait at *k+1* blocks and that gap is exposed on the
critical path.

### 1.1 The asymmetry that motivates this

The **non-scheduler** MFSDP v2 path already supports multi-module prefetch, via a
parameter-element **budget**:

```python
# experimental/module.py
def _prefetch_parameter_groups(self, order, prefetch_size, orientation=BOTH):
    next_module = order.next_item(self)
    if prefetch_size is None:                      # budget disabled
        if next_module is not None:
            next_module._unshard_parameter_groups(orientation)
        return
    prefetched_size = 0
    while next_module is not None and prefetched_size < prefetch_size:
        next_module._unshard_parameter_groups(orientation)
        prefetched_size += next_module.num_parameter_elements
        next_module = order.next_item(next_module)
```

wired from `mcore_fsdp_adapter.py`:

```python
schedule_policy = SchedulePolicy(
    forward_prefetch_size=ddp_config.suggested_communication_unit_size,
    backward_prefetch_size=ddp_config.suggested_communication_unit_size,
)
```

i.e. the budget is the existing CLI knob `--suggested-communication-unit-size`
(`megatron/training/arguments.py`, default `None` → prefetch exactly one successor).

**But that machinery is bypassed on the scheduler path.** `issue_unshard` calls
`module.unshard(orientation=orientation)` with `prefetch` left at its `"none"`
default, so `_prefetch_parameter_groups` never runs. The trace-and-replay path
therefore has **strictly less pipelining than the path it replaces** — it inherits a
depth of one where the module path can go deeper.

This is the gap to close. The plan is also better placed to close it: it knows the
exact sequence of upcoming unshards and each one's orientation, whereas the
module-level walk can only follow a static `forward_order` / `backward_order`.

---

## 2. Goals and non-goals

**Goals**

- Let the scheduler keep more than one gather in flight, with depth bounded by a
  predictable, user-controllable budget.
- Reuse the existing budget semantics and knob rather than inventing a second notion
  of prefetch sizing.
- Make the memory cost explicit and measurable, since extra lookahead directly
  increases resident parameter storage.

**Non-goals**

- Changing the op order, the plan's reshard-skip decisions, or any numerics.
- Adaptive/latency-modelled depth (see §7, future work).
- Touching the non-scheduler path, which already behaves this way.

---

## 3. Design

### 3.1 Chosen option: element budget, plan-driven

Replace the single `prefetch_after` pair with a **prefetch set** built by walking the
plan forward, accumulating `num_parameter_elements` until the budget is reached:

```python
@dataclass
class PlanOp:
    ...
    prefetch_after: tuple[tuple[FsdpModule, str | None], ...] | None = None
```

```python
# _build_plan, step 3 (rewritten)
budget = self._prefetch_budget            # elements; None => a single successor
for i in range(n):
    if events[i].kind is not OpKind.WAIT_UNSHARD:
        continue
    targets, accumulated, seen = [], 0, {events[i].module}
    for j in range(i + 1, n):
        if events[j].kind is not OpKind.ISSUE_UNSHARD:
            continue
        m = events[j].module
        if m in seen:
            continue                       # one lookahead entry per module
        seen.add(m)
        targets.append((m, plan[j].orientation))
        accumulated += m.num_parameter_elements
        if budget is None or accumulated >= budget:
            break
    if targets:
        plan[i].prefetch_after = tuple(targets)
```

and at execution:

```python
def wait_unshard(self, module):
    plan_op = self._record(OpKind.WAIT_UNSHARD, module, None)
    module.wait_unshard()
    if plan_op is not None and plan_op.prefetch_after:
        for target, orientation in plan_op.prefetch_after:
            self._prefetch(target, orientation)
```

`_prefetch` itself is unchanged (it issues `module.unshard(orientation=...)`, which
does not consume an op number).

The budget is sourced from the same place as the module path, so both behave
consistently:

```python
scheduler = TraceAndReplayScheduler(context, prefetch_budget=schedule_policy.forward_prefetch_size)
```

(`forward_prefetch_size` and `backward_prefetch_size` are both set from
`ddp_config.suggested_communication_unit_size`, so one budget covers both phases.)

### 3.2 Why a budget rather than a fixed module count

| option | pro | con |
|---|---|---|
| **A. fixed depth N** | trivial to reason about | N modules of wildly different sizes ⇒ unpredictable memory; needs its own knob |
| **B. element budget** (chosen) | reuses an existing, already-tuned knob; extra residency is bounded and predictable in elements; identical semantics to the module path | depth varies with module size (a feature, not a bug) |
| **C. byte cap** | most precise under MXFP8, where orientation changes bytes | needs orientation-aware byte accounting inside `_build_plan`; more machinery |
| **D. adaptive depth** | potentially optimal | needs a latency model and online feedback; see §7 |

Option **B** is chosen because the dominant risk is memory, and an element budget is
the cheapest way to bound it with a concept the codebase already has.

### 3.3 Ordering and stream semantics — the subtle part

All prefetch work goes onto a single `context.allgather_stream`
(`_unshard_parameter_groups` enters `torch.cuda.stream(allgather_stream)`). So a
deeper prefetch does **not** make gathers parallel; it makes the stream's queue
deeper, so more gathers complete earlier relative to the compute that needs them.

Consequences to respect:

- **More depth is not monotonically better.** Because the queue is FIFO, a demand
  gather issued later can be delayed behind prefetches. Over-prefetching can *raise*
  the stall it was meant to remove, and it certainly raises peak memory. Hence a
  budget, and hence the measurement in §6 must produce a **frontier**, not a single
  "best" value.
- **No recursion.** Prefetch calls `unshard` with `prefetch="none"`, so it cannot
  re-enter the module-level budget walk. This must stay true, or prefetch becomes
  recursive and the two mechanisms double up.
- **Idempotence.** `unshard` returns early when what is resident already satisfies
  the request, so a module prefetched twice (e.g. forward and backward lookahead
  overlapping) is harmless.

### 3.4 Memory model

Peak resident parameter storage becomes roughly

```
peak ≈ (residency of the plan's current window) + (materialization of the lookahead set)
```

The second term is what the budget bounds, and it is *new* memory rather than a
longer hold on existing memory — prefetch extends residency **earlier**, not longer.

Two amplifiers to keep in mind:

- The plan's `orientation` for a lookahead module is the **union over its residency
  window**, so a shared forward/backward window prefetches `BOTH`. That is the same
  effect that cost **+4965 MB** on the plan-widening variant of the orientation fix.
  Multi-module prefetch multiplies it by the depth — the single biggest risk here.
- At the target configuration the run peaks at roughly **150 GB of 288 GB** device,
  leaving of order 138 GB of headroom. Prefetch is a way to *spend* that headroom on
  throughput; the design should be explicit about how much it is willing to spend.

### 3.5 Interaction with the reshard-skip logic

The plan's reshard-skip decides when a module's storage is released
(`need <= resident`). Prefetch does not change those decisions: it only moves a
module's materialization earlier. Two consequences to handle explicitly:

- **Wasted prefetch.** If the plan reshards module *M* (`skip=False`) before *M*'s
  demand unshard, a prefetch of *M* is thrown away. The plan knows where the reshards
  are, so `_build_plan` should drop lookahead entries that a reshard intervenes on —
  cheap to do while walking, and it avoids paying for gathers that cannot be used.
- **Residency accounting.** `_materialized_orientation` / `_unshard_event` are
  per-module and are cleared on reshard, so widening cannot be fooled by stale state.
  No change needed, but the peak-memory claim must be measured, not argued.

---

## 4. Correctness argument

Multi-module prefetch **cannot change numerics**, which is what makes this an
attractive lever:

- Prefetch is **out-of-band**: `_prefetch` does not consume an op number, does not
  alter the recorded events, and adds nothing to the plan's validated op sequence.
- The demand path is unchanged: `_record(OpKind.ISSUE_UNSHARD, ...)` still validates
  replay against the plan, and `module.unshard(orientation=...)` still materializes
  the requested orientation.
- `FsdpModule._unshard_parameter_groups` remains the correctness authority: a request
  already satisfied is a no-op, and a request for a direction that is missing
  **widens in place**, gathering only what is missing. So a prefetch that guessed the
  wrong orientation costs bytes, never correctness.
- Therefore this change is a **pure performance/memory lever**, gated on memory and
  throughput rather than on correctness. The `--deterministic-mode` bitwise
  comparison should still be run because gather *timing* changes, but no value should
  move.

---

## 5. Implementation sketch

| file | change |
|---|---|
| `experimental/schedule.py` | `PlanOp.prefetch_after` becomes a tuple of `(module, orientation)`; `_build_plan` step 3 rewritten to accumulate a budgeted set and to drop entries cut off by a reshard; `wait_unshard` loops over the set; `TraceAndReplayScheduler.__init__` takes the budget |
| `experimental/fully_shard.py` | pass the budget from the schedule policy when constructing the scheduler |
| `experimental/module.py` | no change (prefetch stays `prefetch="none"` here; worth asserting to prevent recursion) |
| tests | extend the `_build_plan` fuzz harness: assert the lookahead set is ordered, distinct, budget-bounded, excludes the module being waited on, and contains no entry whose demand unshard is separated from it by a reshard |
| config | none needed if `--suggested-communication-unit-size` is reused — but note it also feeds `suggested_RS_queue_capacity` in `megatron_fsdp.py`, so attribution must account for that second effect |

---

## 6. Measurement plan

**Step 0 — is there headroom at all?** Before writing code, quantify the exposed
stall from profiles we already have: for each `wait_unshard`, the gap between the end
of the preceding compute and the completion of the all-gather it waits on. If that is
≈ 0, depth 1 is already sufficient and this design should be **abandoned** rather
than implemented. This is the cheapest possible kill criterion and should be run
first.

**Step 1 — budget sweep.** Sweep the budget (at least `None` = depth 1 as control,
`1x`, `2x`, `4x` the current unit size) at the target configuration (32×GB300,
seq 12288 / GBS 128 / MBS 1, 110 iterations) and report the **throughput vs
peak-memory frontier**: median TFLOP/s/GPU over iterations 21–110 against peak
`max_allocated` / device.

**Step 2 — mechanism.** nsys on the best point and the control, same window (rank 0,
steps 20–30): `AllGather_RING_LL` time and launch count, `SendRecv`, **and their
sum**; the pipeline fill+drain envelope; and specifically the **wait-stall time** from
step 0, to show the stall actually shrank.

**Step 3 — guardrails.** `--deterministic-mode` bitwise loss and grad-norm parity;
activation proof; 0 NaN/skip/errors.

For context when interpreting the frontier: we have measured that roughly 62–69 GB of
extra memory buys about 5.4–9.1% throughput moving from this stack (~150 GB) to the
DDP + layer-wise distributed-optimizer path (~213 GB). Prefetch is the same trade in
the opposite direction — spending memory we already have to buy back some throughput,
without giving up parameter sharding.

---

## 7. Open questions and future work

1. **Optimal depth is workload-dependent** — a function of microbatch count, module
   sizes and gather latency. Worth reporting the frontier rather than a single number,
   and re-checking if the topology changes.
2. **Adaptive depth** (option D): derive the lookahead from measured gather latency
   versus module compute time. The plan plus per-op timings from a tracing step
   already contain most of the inputs. Deferrable.
3. **Byte-aware budget** (option C) would let an MXFP8 shared window charge `BOTH` at
   twice the rate of a single orientation, which is closer to the real cost than an
   element count. A natural follow-up if the frontier shows the element budget
   mispricing orientation.
4. **Prefetch across a reshard** — currently dropped (§3.5). If the plan's skip
   heuristic later releases storage less often, more lookahead becomes usable.
5. **Does the union orientation in `prefetch_after` still make sense at depth > 1?**
   For a deep lookahead, prefetching a later module's window union early may be
   wasteful; prefetching only its *first* orientation and letting the module widen on
   demand may dominate. Worth measuring as a variant.
