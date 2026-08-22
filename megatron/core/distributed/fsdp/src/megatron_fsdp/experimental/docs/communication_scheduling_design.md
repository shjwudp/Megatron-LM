# M-FSDP v2 trace-guided communication scheduling

## Status and scope

This document describes the initial opt-in communication scheduler for experimental
M-FSDP v2. The scheduler uses the existing execution trace to delay parameter
all-gather (AG) prefetch and gradient reduce-scatter (RS) until user-declared
module execution points. Its first target is VPP + combined 1F1B with expert
parallel (EP) communication, but the API is intentionally independent of a
particular EP backend.

The public policy/configuration API, occurrence recording, delayed and
depth-adjustable AG prefetch, deferred RS, automatic byte-budget inference,
MCore selector translation, and combined-1F1B named anchors are implemented.
The feature remains experimental and disabled unless scheduling rules are
supplied.

Related documents:

- `1f1b_ep_overlap_design.md` describes fine-grained FSDP hooks and
  occurrence-order replay.
- `trace_pool_allocator_design.md` describes trace-planned temporary-buffer
  storage.

## Motivation

M-FSDP v2 currently launches the traced successor's AG immediately after
unsharding the current FSDP unit. It also packs and launches an RS immediately
when a unit's gradients become ready. Eager launch normally maximizes
communication/computation overlap, but it can be counterproductive when FSDP
and an external communication library share GPU communication resources.

In the motivating PP3/VPP2/EP8 profile, an EP combine that takes about
0.40 ms without concurrent FSDP communication takes about 0.87--1.08 ms when
it overlaps an FSDP AG. Depending on the pipeline rank, 26--40% of AG GPU time
overlaps EP communication. The non-overlapped M-FSDP combine time matches the
ND-parallel baseline, which makes communication contention the primary
hypothesis.

M-FSDP cannot generally discover that an arbitrary submodule launches
communication. `torch.distributed` interception would miss custom CUDA
extensions such as HybridEP, and a Python module post-hook does not imply that
asynchronous GPU work has completed. Instead, users or schedule integrations
declare execution points after which FSDP may launch delayed communication.

## Goals

1. Let a user delay a future FSDP prefetch until a configured descendant
   submodule has executed, and choose its occurrence lookahead depth.
2. Let a user nominate modules whose pre-backward entry releases pending RS
   requests.
3. Infer the pending-RS byte budget by default while retaining an advanced
   explicit override.
4. Enforce the pending budget at every RS-input allocation, including fused
   weight-gradient allocation in pre-backward.
5. Preserve VPP occurrence ordering, collective ordering, gradient
   accumulation semantics, and trace-pool storage safety.
6. Degrade to immediate communication when a hint cannot be honored safely.

## Non-goals

- Automatically classify arbitrary CUDA work as communication.
- Provide bandwidth quality-of-service or preemption for already-running
  communication kernels.
- Make timing-dependent, per-rank launch decisions during replay.
- Guarantee that a release point is contention-free. A release point is a
  lower bound on launch time; the user chooses it using model knowledge.
- Enable CUDA graphs. M-FSDP v2 currently rejects CUDA-graph execution, and a
  later design must make the compiled stream/event topology capture-stable.

## Terminology

- **FSDP unit**: one `FsdpModule` and its owned parameter groups.
- **Occurrence**: one runtime execution of an FSDP unit or execution point.
  VPP can execute the same object many times in one global batch.
- **Demand AG**: materialization required by the current consumer. Demand AG
  is never delayed by policy.
- **Future AG**: speculative prefetch for the configured-depth future traced
  `UNSHARD` occurrence. Depth one is the immediate successor.
- **Completion anchor**: a configured submodule post-forward or post-backward
  occurrence. A CUDA event recorded on the submodule's execution stream
  represents completion.
- **RS release module**: a configured module whose pre-backward occurrence may
  launch a previously prepared RS request.
- **Pending RS bytes**: RS-input storage admitted for deferral whose collective
  has not been submitted. A request that cannot enter the deferral budget is
  marked submit-on-ready and does not become a queued request.
- **In-flight RS bytes**: RS-input storage whose collective has been submitted
  but whose stream-ordered lifetime has not completed.

The design deliberately avoids the term *safe point*. A completion anchor is
correctly ordered after one operation, but it does not prove that a later
communication operation cannot overlap the launched FSDP collective.

## Baseline behavior without scheduling rules

Without a communication scheduler, the execution runner observes:

```text
UNSHARD(module, orientation)
RESHARD(module)
```

During replay, `_unshard_and_prefetch()` asks the runner for the next traced
`UNSHARD` and launches that AG immediately:

```text
UNSHARD(A) -> AG(B) -> compute(A)
```

`post_backward()` currently performs:

```text
gradients ready -> allocate/obtain partial-grad buffer -> pack -> RS -> release
```

The trace-pool allocator observes allocation lifetimes through the initial
execution trace and one prefetch-enabled replay, then builds fixed slots. A
delayed-communication replay must therefore run before trace-pool planning so
the extended AG and RS-input lifetimes are visible to the allocator.

## Proposed API

Scheduling has context-wide state and per-FSDP-unit annotations. The low-level
API separates those concerns.

```python
from dataclasses import dataclass
from typing import Literal

from torch import nn


@dataclass(frozen=True)
class ModuleCompletion:
    module: nn.Module
    phase: Literal["forward", "backward"]


@dataclass(frozen=True)
class NamedCompletion:
    name: str
    phase: Literal["forward", "backward"]


@dataclass(frozen=True)
class NamedPreBackward:
    name: str


@dataclass(frozen=True)
class FsdpModuleCommunicationPolicy:
    # Release this unit's traced successor prefetch after one of these
    # descendant completion anchors.
    prefetch_successor_after: tuple[ModuleCompletion | NamedCompletion, ...] = ()

    # A pre-backward occurrence of one of these descendants may release an
    # RS request from the context-wide pending queue.
    reduce_scatter_release_on_pre_backward: tuple[nn.Module | NamedPreBackward, ...] = ()


@dataclass(frozen=True)
class FsdpCommunicationSchedulerConfig:
    # None: infer after the trace. Zero: do not defer RS. Positive: explicit
    # pending-byte override.
    max_pending_reduce_scatter_bytes: int | None = None

    # One selects the immediate successor. N selects the Nth future traced
    # UNSHARD occurrence and trades additional parameter residency for lead.
    prefetch_depth: int = 1
```

The configuration is passed once to the shared context, and annotations are
passed while wrapping each FSDP unit:

```python
with fully_shard_context(
    ...,
    communication_scheduler=FsdpCommunicationSchedulerConfig(),
):
    fully_shard(
        layer,
        ...,
        communication_policy=FsdpModuleCommunicationPolicy(
            prefetch_successor_after=(
                ModuleCompletion(layer.prefetch_release_module, "forward"),
            ),
            reduce_scatter_release_on_pre_backward=(
                layer.rs_release_module,
            ),
        ),
    )
```

A non-empty communication policy requires occurrence tracing. Supplying a
communication scheduler therefore enables `use_trace_replay`; users should not
need to coordinate two independent switches.

All VPP chunks must join the same `FsdpContext` and use an equal scheduler
configuration. `fully_shard_context(reuse_existing=True, ...)` rejects a chunk
whose scheduler configuration differs from the ambient context. The outermost
construction scope owns finalization; joining chunks neither finalize nor
replace scheduler state.

### API semantics

- `prefetch_successor_after` belongs to the **source** FSDP unit. It means
  "release this occurrence's configured-depth future prefetch after this
  descendant completes." The API name retains *successor* for compatibility;
  `prefetch_depth=1` is the immediate successor. It does not mean gathering an
  FSDP unit after one of its own parameter-consuming descendants, which would
  be circular.
- `prefetch_depth=N` selects the Nth future `UNSHARD` occurrence in the shared
  context trace, counting repeated VPP executions independently and wrapping
  at the global-batch boundary. A single context-wide depth produces a
  one-to-one cyclic shift: every occurrence has exactly one speculative
  producer and target.
- Increasing depth moves the target farther into the future but does not move
  the configured launch anchor. It therefore creates more lead after the
  protected communication at the cost of more simultaneously resident full
  parameters. Depth must be positive and cannot exceed the number of traced
  `UNSHARD` occurrences.
- A completion anchor must be a descendant of the annotated FSDP unit and
  must occur after that unit's `UNSHARD` and before the target's demand
  `UNSHARD` in the trace.
- If several configured anchors match one occurrence, the first matching
  anchor observed at runtime releases the queued successor.
- `reduce_scatter_release_on_pre_backward` marks release modules; it does not
  identify which unit's RS to launch. The compiled occurrence trace assigns
  the oldest legal pending RS request to each release occurrence.
- An empty module policy preserves eager behavior.
- `max_pending_reduce_scatter_bytes=None` is the normal user experience and
  requests automatic inference. `0` disables RS deferral without disabling
  delayed AG. A positive value is an expert override.
- The scheduler aligns a positive override to the greatest common divisor of
  traced request sizes and logs the effective value. If no request fits, RS
  remains eager.
- A module completion hook records the current CUDA stream. A module that
  launches work on a private stream must join that work before returning, or a
  schedule integration must provide the actual completion event explicitly.
- Collective domains come from the `DeviceMesh` already carried by each
  parameter group. The scheduler stores those process groups on its requests
  and does not discover groups through MCore `parallel_state` globals.

### MCore integration

The low-level API uses module objects and does not parse strings. The MCore
adapter may expose module-type or root-relative-FQN selectors, resolve them
against `named_modules()` before wrapping, and construct the low-level policy.
The adapter must fail on unmatched selectors. Selectors intended to match
multiple modules must use explicit type or glob semantics and log every match.

The experimental MCore CLI surface is:

```text
--fsdp-prefetch-successor-after SOURCE_GLOB:PHASE:DESCENDANT_GLOB
--fsdp-prefetch-depth N
--fsdp-reduce-scatter-release-on-pre-backward SOURCE_GLOB:DESCENDANT_GLOB
--fsdp-max-pending-reduce-scatter-bytes:
  None for auto, 0 for eager RS, or a positive byte override
```

Both rule arguments may be repeated. `PHASE` is `forward` or `backward`.
Prefix `DESCENDANT_GLOB` with `@` to select a named combined-schedule node;
otherwise it is matched against root-relative `named_modules()` names inside
the selected FSDP unit. `<root>` and `<self>` name the model root and selected
unit itself. Construction fails when a source or descendant selector is
unmatched.

Selectors belong in the adapter rather than the transport implementation.
This keeps `megatron_fsdp.experimental` independent of Transformer-layer names
and lets programmatic callers use module objects without a string grammar.

The combined 1F1B schedule already exposes semantic `moe_dispatch` and
`moe_combine` schedule nodes. Those nodes are not necessarily `nn.Module`
objects, so the adapter emits the equivalent runner anchor directly at the
node boundary. It does so through an adapter-facing context method rather
than reaching into `FsdpExecutionRunner`:

```python
context.record_completion_anchor(
    owner=layer,
    anchor="moe_combine",
    phase="forward",
    event=node.event,
)
```

`owner`, `name`, and `phase` form a stable logical key; `event` is freshly
recorded by the node on its actual stream for each occurrence. This method is
an integration escape hatch for execution engines that bypass module hooks,
not a general claim that FSDP detected communication. The scheduler itself
does not depend on HybridEP, NCCL-EP, DeepEP, or MoE.

## Trace event model

The reusable execution trace is extended with semantic ordering events:

```text
UNSHARD(fsdp_module, orientation)
RESHARD(fsdp_module)
COMPLETION(owner, anchor, phase)
RS_RELEASE(owner, anchor)
```

RS reserve/readiness/flush operations are recorded by a context-local budget
simulator rather than inserted into the runner's successor trace. The
simulator retains request size and collective-domain metadata needed to infer
the replay budget.

CUDA events are runtime objects and are not stored as part of the reusable
trace identity. During replay, the matching hook records a fresh event on the
actual execution stream and passes it to the launch path.

The trace remains occurrence-based. Object identity selects a configured
rule, while the trace index distinguishes repeated VPP/microbatch
occurrences.

At trace completion, each rank independently freezes the occurrence trace.
The inferred budget is reduced with `MIN` through the explicit process groups
carried by each request's `DeviceMesh`, so all ranks in a collective domain use
the same capacity. Cross-rank trace-signature validation is useful future
hardening; today collective-order divergence remains subject to the same
SPMD requirement as the underlying FSDP execution.

## Delayed depth-adjustable all-gather

### Trace

The first global batch continues to run without speculative prefetch. The
runner records both the real `UNSHARD` order and configured completion-anchor
occurrences. During replay, the configured depth selects the Nth future
`UNSHARD` target and its orientation. When the source policy contains an anchor
for the current forward/backward orientation, the scheduler queues that target
instead of submitting it immediately. If no matching anchor occurs before
demand, the consumer submits the AG itself.

### Replay

```text
UNSHARD(A), prefetch_depth=N
  -> identify T as the Nth future UNSHARD occurrence
  -> queue AG(T), do not submit it

ANCHOR_DONE(A.x)
  -> record completion event on A.x's execution stream
  -> allgather_stream.wait_event(anchor_event)
  -> submit AG(T)
```

At `UNSHARD(T)`, demand execution remains the correctness backstop. If T's AG
was never released, it is submitted immediately. If it was released but has
not completed, the consumer waits on T's existing unshard event. Both cases
increment separate diagnostics.

A completion anchor is the earliest legal launch point: it only orders AG
after previous work. It cannot guarantee completion before later EP work, and
adding a `before` dependency would either be advisory or make the later work
wait for AG. The intended tuning rule is therefore to place the anchor after
the protected EP communication and increase `prefetch_depth` only far enough
to hide AG behind subsequent compute. Nsight validation remains necessary
because stream ordering alone cannot provide bandwidth quality-of-service.

## Deferred reduce-scatter

### Request state machine

```text
UNALLOCATED
    |
    | reserve RS-input bytes
    v
WRITING -- gradients complete --> READY -- submit --> IN_FLIGHT -- retire --> DONE
```

An RS request stores:

- parameter group and collective domain;
- packed `DBuffer` or fused-wgrad buffer;
- byte count;
- readiness event from its producing compute stream;
- `is_last_microbatch` captured when the gradients become ready;
- trace occurrence and diagnostics metadata.

Capturing `is_last_microbatch` is required. A deferred request may be launched
from a later microbatch's pre-backward scope, where the context's current
`no_sync` state may differ from the state that produced the gradients.

### Release on configured pre-backward

When a configured release module enters pre-backward, the scheduler performs
the following ordering:

1. Ensure the release module's demand AG has been submitted.
2. Make the RS stream wait for that demand-AG completion event when both use
   contending resources.
3. Submit the FIFO RS request assigned to this release occurrence.
4. Let the release module's backward compute overlap the RS.
5. Keep successor AG governed by its independent completion-anchor policy.

A release hook does not drain the queue. It submits at most one ready,
deferred request: the oldest request that is also the FIFO head of its
collective domain. A release reached before any request is ready is a no-op;
capacity enforcement or the final flush will submit the request later. This
keeps the hint advisory and avoids extending a burst across several protected
communication phases.

The implementation keeps global creation order for diagnostics and a
separate byte budget/FIFO constraint per `DeviceMesh` domain. Disjoint
domains may progress independently, while requests within one domain never
overtake each other.

### Hard completion deadline

All remaining requests are submitted and the consumer stream waits for the RS
stream in `finish_grad_sync()`. This is the correctness deadline before
gradient finalization, clipping, norm calculation, and optimizer access.

The operation is context-wide and idempotent after the queue is drained. VPP
exposes multiple model chunks that share one context, and
`finalize_model_grads()` calls `finish_grad_sync()` on each chunk. The first
call drains the queue; later calls only preserve the stream dependency.

`FsdpContext.complete_trace()` is not a correctness flush point: the current
`FullyShardedOptimizer` calls it after the optimizer step. It remains only the
global-batch trace/planning boundary.

## Automatic pending-byte inference

Users should not need to calculate a byte value. During the first immediate-RS
trace, the scheduler records exact request sizes and simulates the configured
release policy.

The inferred budget is:

```text
required_peak = peak pending bytes in the simulated release plan
available_extra = observed minimum device headroom - safety reserve
candidate = min(required_peak, available_extra)
effective_budget = largest request-aligned budget <= candidate
```

Properties:

- Request size is computed from the partial-gradient layout and communication
  dtype before allocation; it is not estimated from parameter dtype.
- Device headroom is sampled only during trace. Replay never uses current free
  memory to make a rank-local launch decision.
- The effective budget is reduced to the minimum supported value across ranks
  in each participating collective domain, then frozen for replay.
- The safety reserve protects non-FSDP temporaries and trace/replay variation.
  Its value is logged and may become a separate advanced configuration later.
- If no complete RS request fits, effective budget is zero and RS remains
  eager.
- The initial trace-pool plan does not yet exist when the budget is inferred,
  so the first run uses conservative device headroom and does not increase the
  budget after pool planning.

The scheduler logs required peak, observed headroom, safety reserve, user
override, and effective aligned budget.

## Allocation-pressure fallback

Capacity is checked **before** every full RS-input allocation:

```python
incoming_bytes = parameter_group.partial_grad_nbytes()

while pending_bytes + incoming_bytes > effective_budget and has_ready_request():
    submit_oldest_ready_request(reason="capacity")

if pending_bytes + incoming_bytes <= effective_budget:
    reserve_for_deferral(incoming_bytes)
else:
    mark_submit_on_ready(incoming_bytes)

allocate_rs_input()
```

This check covers two allocation sites:

1. ordinary gradients allocate a packed partial-gradient buffer after their
   gradients become ready;
2. fused weight-gradient accumulation allocates its full staging buffer in
   pre-backward, before TE writes the weight gradient.

An incoming request larger than the effective budget is still allocated for
the backward computation, but it is marked submit-on-ready and never waits in
the deferred queue. If the oldest existing request is not ready, it cannot be
submitted; the incoming request is likewise submit-on-ready. A failed physical
allocation cancels its scheduler reservation and propagates the allocator
error without leaving stale pending-byte state.

Submitting an RS removes its bytes from the *pending* budget, but the storage
may remain physically resident until stream completion. The physical allocator
therefore tracks queued and in-flight lifetimes separately. A launch alone is
not proof that caching-allocator memory is immediately reusable.

Capacity-triggered submissions preserve FIFO collective order and increment a
`forced_capacity_flushes` diagnostic. Because the effective budget and request
sequence are rank-consistent, the same forced submissions occur on every rank
in a collective domain.

## Trace-pool integration

Delayed communication extends buffer lifetimes and must participate in trace
pool planning:

- A queued successor AG contains only target metadata. It does not allocate
  the gather output until its completion anchor releases it. The shifted
  allocation-to-reshard lifetime is then recorded normally.
- A trace-pooled partial-gradient key remains active from allocation through
  pending time. It is freed only after its RS has been enqueued on the ordered
  reduce-scatter arena.
- The first replay enables delayed AG and RS. Trace-pool planning occurs only
  after that replay, matching the allocator's existing prefetch-aware
  lifecycle.
- The current trace pool uses one logical partial-gradient key per parameter
  group. Before another occurrence of the same group allocates that key, the
  older request must be submitted and stream-ordered for reuse. Supporting
  multiple simultaneously pending occurrences of one group would require
  generational keys and is not part of the first implementation.
- Fused-wgrad staging is allocated on the compute stream and is not currently
  a trace-pool partial-gradient allocation. Its reservation still counts
  toward the scheduler budget, and its storage follows `record_stream`
  lifetime rules after RS submission.

The trace pool may back its slots with PyTorch NCCL symmetric memory. Legacy
FSDP double buffering remains incompatible with the trace pool; Megatron-FSDP
v2 therefore does not enable the v1 manual-registration or double-buffer
defaults when `--use-nccl-ub` is selected.

## Failure and fallback behavior

### Missing or late anchor

If the trace does not contain a configured completion anchor before the target
consumer, the corresponding AG uses demand/eager behavior. The scheduler logs
the source, target, phase, and missing selector.

### Replay divergence

On execution-trace divergence:

1. stop issuing speculative delayed AG;
2. submit pending RS requests in recorded FIFO order;
3. revert new requests to eager communication;
4. retrace from the divergence event;
5. do not reuse a trace-pool slot if its lifetime conflicts with the optimized
   plan.

This fallback is allowed only while every collective domain can preserve its
already-established FIFO sequence. Divergence that changes the collective
sequence, membership, dtype, or element count is fatal. A trace-pool lifetime
collision that cannot be resolved by waiting for the recorded owning event is
also fatal; the scheduler must not guess at rank-local storage reuse.

### Collective-order mismatch

A runtime request-order mismatch is an error, not a performance fallback.
Continuing could deadlock NCCL or silently mix gradient reductions. Explicit
cross-rank trace-signature checking is left as follow-up hardening.

### Memory pressure

Budget pressure first forces FIFO RS submission. Physical allocator OOMs are
reported normally after the reservation is rolled back.

## Diagnostics

At trace compilation the runner logs trace length and replay state, while the
RS scheduler logs required peak, inferred/overridden context limit, observed
free/total device bytes, and effective budget per domain. Periodic scheduler
reports include delayed AG count, anchor and demand releases, pending AGs,
RS anchor/capacity/final releases, and current pending RS bytes. Selector
resolution logs each matched FSDP unit and anchor.

The scheduler emits launch-scoped NVTX ranges around every parameter-group
collective submission. AG labels include the target FSDP module, parameter-group
index, payload orientation, and release path (for example, `anchor`, `demand`,
or `consumer`). RS labels include the owning FSDP module, parameter-group index,
and release reason (for example, `anchor`, `capacity`, `submit-on-ready`, or
`finish_grad_sync`). Keeping the CUDA launch API inside the range is required
for Nsight to associate a later asynchronous kernel with the scheduling
decision that submitted it; an instantaneous marker after submission is not
sufficient, especially when autograd runs the hook on a worker thread.

## Validation plan

### Unit tests

The initial implementation adds tests under
`tests/unit_tests/distributed/mfsdp_v2/` for:

1. scheduler/config validation and shared-context compatibility;
2. completion-anchor release, demand fallback, and depth-based future target
   selection across trace wrap and repeated VPP module occurrences;
3. deferred RS release, automatic budget compilation, and captured
   `is_last_microbatch`;
4. multi-step loss parity against eager execution;
5. MCore module/named-selector translation.

Follow-up coverage should exercise repeated VPP occurrences, replay
divergence with fused-wgrad buffers, capacity-forced release, trace-pool reuse,
and multiple collective domains directly.

Distributed unit tests must run through `torch.distributed.run`; single-process
tests are insufficient for collective ordering.

### Functional correctness

Run multi-step loss/gradient parity against eager M-FSDP v2 for:

- PP3/VPP2/EP8 combined 1F1B;
- dense and expert parameter groups;
- multiple microbatches with `no_sync`;
- MXFP8 fused wgrad;
- dense HFSDP inner-DP sharding;
- trace pool enabled and disabled.

The functional run must show identical collective counts/order per domain and
no permanent replay divergence.

### Performance

Compare identical 24-GPU real-data jobs for:

1. eager M-FSDP v2;
2. delayed AG at depth one;
3. delayed AG at successively larger depths until exposed AG stops improving
   or parameter residency becomes unacceptable;
4. the best delayed AG depth plus pre-backward RS release;
5. ND-parallel reference.

Discard trace/planning steps and at least the first five steady-state steps.
Report mean, median, and standard deviation for step time and model TFLOP/s;
peak allocated, reserved, and device-used memory; AG/RS exposed waits; and EP
dispatch/combine durations conditioned on FSDP overlap. Capture at least one
rank from every pipeline stage.

## Alternatives considered

### Automatically detect arbitrary communication

Rejected as the core API. Framework-level collective interception misses
custom CUDA extensions, while CUPTI/Nsight discovery is too heavyweight and
post-hoc for the runtime scheduler.

### Require users to enter the RS byte budget

Rejected as the default. Users know semantic release modules more readily than
temporary-buffer byte sizes, dtypes, and allocator headroom. A positive-byte
override remains useful for experiments.

### Delay every RS until the global-batch boundary

Rejected. It maximizes unreduced-gradient residency, loses backward overlap,
and can exceed trace-pool capacity. `finish_grad_sync()` is a deadline, not the
normal launch policy.

### Use CUDA stream priority only

Rejected as a deterministic solution. Stream priority affects scheduling of
eligible kernels but does not provide fabric bandwidth isolation or preempt an
already-running NCCL kernel.

### Configure `prefetch_after` on the target module

Rejected because the phrase is ambiguous and can describe a circular
dependency on a target-owned, parameter-consuming descendant. The API uses
`prefetch_successor_after` on the source FSDP unit instead.

## Implementation status

Implemented in this change:

1. policy dataclasses, validation, and occurrence trace events;
2. delayed depth-adjustable AG with anchor and demand release;
3. per-domain deferred-RS queues, pre-backward release, and final flush;
4. automatic/explicit pending-byte budgets and pre-allocation capacity release;
5. trace-pool lifetime preservation through collective submission;
6. MCore selector translation and combined-schedule named anchors;
7. targeted distributed unit tests and loss-parity coverage.

The feature has no default policy. The PP3/VPP2/EP8 performance experiment,
memory-boundary tuning, broader fused-wgrad/trace-pool coverage, and trace
signature hardening remain validation/follow-up work.
