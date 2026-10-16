# Unified MFSDP execution trace prototype

This branch follows #8207 and explores a parameter-lifecycle trace shared by
conventional VPP and combined 1F1B. It is opt-in and needs distributed GPU
validation before production use. The existing pipeline schedules retain
control of compute, P2P, expert communication, and delayed weight gradients.

## Why forward/backward lists are insufficient

VPP revisits a chunk across microbatches. Combined 1F1B also interleaves fine-grained
forward and backward nodes, sometimes demanding the same FSDP owner several times
before releasing its parameters. Separate forward/backward lists omit the actual
interleaving and parameter-release boundaries.

`ExecutionTrace` records one chronological sequence of **host-side demand and
release calls**. Each `ExecutionEvent` contains a weak module reference, an
`unshard`/`reshard` operation, a phase label, and an optional `ExecutionStep`:

```text
ExecutionStep(chunk, microbatch, phase, node)

unshard(A, forward) → unshard(B, backward) → reshard(B)
                   → reshard(A) → unshard(A, forward) → ...
```

The trace has one strict occurrence cursor and no module-to-index lookup. It
checks module identity, operation, phase, and any scheduler labels. Reentering a
root or revisiting a chunk does not reset it.

## Opt-in integration

The iteration scope is owned by the caller/scheduler and must include the entire
schedule: warmup, steady state, cooldown, explicit releases, and delayed wgrad.
The existing schedule entrypoints are not wrapped automatically by this prototype.
For chunks constructed together in one FSDP context:

```python
context = model_chunks[0].module.context
trace = context.execution_trace
key = (forward_only, num_microbatches, microbatch_group_size_per_vp_stage)
with trace.iteration(key=key):
    forward_backward_func(...)
```

Include any other inputs that alter execution order in `key`. The first iteration
for a new key records with speculative prefetch disabled. Subsequent iterations
reset the single cursor at entry and validate complete consumption at exit. Only
the latest successful pattern is cached, so train → validation → train records
again when the key changes. This keeps cache ownership and invalidation explicit.

Both hook paths report through `FsdpModule.unshard()` and `reshard()`:

- Conventional hooks supply forward/backward phases through the existing prefetch
  argument. Recomputation demand calls are recorded too.
- Combined 1F1B's custom hooks supply an explicit execution phase while retaining
  demand-only behavior outside an active trace scope.
- Speculative gathers call `_unshard_parameter_groups()` directly and never become
  new demand occurrences in the trace.
- Repeated backward pre-hooks may keep a shared module in its backward phase until
  its parameter gradients finish accumulating; its phase/NVTX scope opens once.

Schedulers may annotate a step at their own boundary:

```python
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import ExecutionStep

with trace.step(ExecutionStep(chunk=0, microbatch=1, phase='forward', node='attention')):
    forward_node.forward(inputs)
with trace.step(ExecutionStep(chunk=1, microbatch=0, phase='backward', node='mlp')):
    backward_node.backward(output_grad)
```

These optional labels are not automatically populated by the existing schedulers.
For combined execution they belong around individual node invocations; a single
forward label around an entire combined step would mislabel its backward work.
Labels are useful for validation and inspection; recording actual demand/release
calls does not require them. The trace does not replace `ScheduleNode` or
`AbstractSchedulePlan`.

## Release-aware prefetch

During replay, lookahead considers future demands in their actual interleaved
order. A future release of a module blocks its later demands from the current
lookahead: gathering that module now could be invalidated before use. Repeated
demands for one owner are deduplicated, and already-materialized modules do not
consume the prefetch budget. No lookahead crosses the iteration boundary.

Existing all-gather stream waits, materialization events, and release operations
remain responsible for CUDA dependencies. A trace is host launch order, not a
cross-stream completion graph. This prototype does not schedule reductions or
alter gradient-readiness ownership.

## Validation and limits

Focused source-level CPU checks exercise the production automatic/custom hooks
with real autograd and mocked CUDA communication. They verify replayed gradients,
that capture performs no speculative gathers, that replay does prefetch without
adding gathers in the tested schedules, grouped VPP calls, interleaved node labels,
release barriers, weak ownership, and failed/incomplete replay.

Local formatting used the installed Black 24, isort, and Ruff. The repository's
Black 26 check and distributed GPU tests remain pending.

These checks do not establish NCCL ordering, GPU overlap, or peak memory bounds.
Lookahead currently scans future events; its cost on long iteration traces also
needs measurement before selecting a production representation.
All ranks must follow repeatable, collective-compatible patterns. CUDA graphs,
concurrent host execution, and data-dependent schedules are not supported by this
prototype. Failed model execution may leave collectives or module state incomplete;
trace bookkeeping does not make partially executed training recoverable.

Outside an explicit trace scope, #8207's construction-order prefetch remains the
fallback. This allows the new representation and its scheduler integration to be
reviewed separately from changing the default scheduling behavior.
