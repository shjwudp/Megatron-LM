# MFSDP v2 iteration trace prototype

This follow-up to #8207 explores removing `_unique_index` entirely. It is a
review prototype; distributed execution and performance still need GPU validation.

`ExecutionOrder` contains weak references to execution occurrences and a cursor.
`advance()` takes no module argument, performs no lookup, and never wraps.
`FsdpContext` checks that the next occurrence matches the actual module call.
Root membership uses a `WeakKeyDictionary`.

## Recording and replay

One `FsdpContext.iteration()` scope covers an entire schedule, including VPP
warmup, steady state, cooldown, and every microbatch and chunk. The first
successful call records separate forward and backward traces, using demand
all-gathers with speculative prefetch disabled. Later calls reset both cursors
once at entry and replay the recorded occurrences. Exit checks complete trace
consumption. Failed recording never installs a trace.

For example, the actual VPP forward sequence may be `A → a → A → a → B → b → B → b`.
Both occurrences of `A` and `a` are retained. Entering the second root or the next
microbatch does not reset the cursor.

The three standard pipeline schedule entrypoints enter this scope automatically,
once for each distinct MFSDP v2 context. VPP chunks sharing a context therefore
record one trace across roots. Training and forward-only validation have separate
cached traces. A change in microbatch count, chunk count, VPP grouping, model mode,
or relevant schedule inputs starts recording again for that mode.

Standalone callers explicitly define the repeated work unit:

```python
# After fully_shard_context construction has finalized:
with context.iteration(key=(num_microbatches, grouping), forward_only=False):
    # Execute the complete forward/backward schedule here.
    ...
```

Do not additionally wrap the automatically scoped schedule entrypoints. Outside
an iteration scope, execution remains demand-only. The standalone overlap tests
use a single forward/backward microbatch as their repeated unit; VPP needs the
complete scheduler scope.

## Boundaries of this prototype

- All ranks must use repeatable, collective-compatible execution patterns. A
  replay mismatch raises; dynamic control flow needs a new caller-supplied key
  or another trace invalidation policy.
- CUDA graphs and custom combined MoE schedules are rejected by the integration;
  this version requires automatic Python execution hooks.
- Failed work can leave module state or collectives incomplete. Discard failed
  graphs; trace bookkeeping does not recover partially executed training.
- Forward and backward traces are independent. Prefetch can cross microbatch
  boundaries within a phase and be invalidated by an intervening operation in
  the other phase. Validate collective ordering, extra all-gathers, peak memory,
  and overlap on GPUs before considering this production-ready.

The focused tests cover a simulated PP=2/VPP=2 schedule with warmup/1F1B/cooldown,
shared trainable layers, recording/replay, train/validation transitions, changed
VPP grouping, failed capture, replay divergence, and incomplete consumption.
Local checks use extracted production source with CPU autograd and mocked CUDA
communication; they do not establish distributed scheduling correctness.
