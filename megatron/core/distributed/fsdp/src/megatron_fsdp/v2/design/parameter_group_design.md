# Parameter Group Design

## Status

This document defines the target ownership and state model for Megatron FSDP v2
`ParameterGroup`. It is the sole parameter-group implementation used by
`fully_shard()`. It supports eager execution, communication overlap, trace-pool
allocation, per-module and full-iteration CUDA graphs, explicit CPU offload,
delayed backward callbacks, and FP8/NVFP4 parameter gather.
The HSDP lifecycle is defined in [`hsdp_design.md`](hsdp_design.md).

## Design principles

`ParameterGroup` represents parameter state as placement transitions on a
`DeviceMesh`. It does not cache named process groups such as `dp_group` or
`outer_dp_group`. `DataParallelBuffer` obtains the process group for a collective
from `mesh.get_group(changed_axis)` when executing a redistribution.

Sharding-strategy strings are resolved once, before runtime operations, into a
placement-only layout. Runtime code compares source and target placements; it does
not branch on inner or outer group names.

For a 2D HSDP mesh, placement tuples have the fixed axis order
`(outer DP, inner DP)`. Multi-axis communication follows a nesting invariant:

```text
weight all-gather:       outer DP -> inner DP
gradient reduce-scatter: inner DP -> outer DP
```

The inverse ordering is required because the optimizer value is nested-sharded
across the same flattened tensor dimension. An outer shard must first be
all-gathered into the persistent inner-sharded model weight before the inner
shard can be materialized for compute. Backward reverses that construction:
the full local gradient is first reduced into its persistent inner shard, then
the last backward reduces that shard across outer DP for the optimizer.
When axis operations use different CUDA streams, each consumer stream waits for
the preceding producer stream. The exact forward and backward dependency chains
are documented in
[`hsdp_design.md#cross-stream-dependencies`](hsdp_design.md#cross-stream-dependencies).

The group has three persistent distributed values:

- one or more model-weight representations used to materialize compute parameters;
- optimizer main weights;
- persistent accumulated or reduced gradients.

It may temporarily lease a full model-weight buffer. Strategies that reduce
every microbatch also lease a full gradient buffer. DDP and ZeRO-1 instead use
their persistent replicated gradient storage directly. These leases and views
are logical runtime state, not persistent buffer roles.

## Placement layout

```python
@dataclass(frozen=True)
class ParameterGroupLayout:
    weight: tuple[Placement, ...]
    main_weight: tuple[Placement, ...]
    grad_storage: tuple[Placement, ...]
    grad_accumulation: tuple[Placement, ...]
```

Two layouts are derived from the mesh:

```python
full = (Placement.REPLICATE,) * mesh.ndim
contribution = (Placement.PARTIAL,) * mesh.ndim
```

The optimizer gradient has the same placements as `main_weight`. The
microbatch accumulation placement separates reductions performed every
backward from reductions delayed until the last backward:

| Strategy | Weight | Main weight | Gradient storage | Microbatch accumulation | Final gradient |
| --- | --- | --- | --- | --- | --- |
| DDP (`no_shard`) | `[R]` | `[R]` | `[R]` | `[P]` | `[R]` |
| ZeRO-1 (`optim`) | `[R]` | `[S]` | `[R]` | `[P]` | `[S]` |
| ZeRO-2 (`optim_grads`) | `[R]` | `[S]` | `[S]` | `[S]` | `[S]` |
| FSDP (`optim_grads_params`) | `[S]` | `[S]` | `[S]` | `[S]` | `[S]` |
| HSDP, outer replicated | `[R,S]` | `[R,S]` | `[R,S]` | `[P,S]` | `[R,S]` |
| HSDP, outer sharded | `[R,S]` | `[S,S]` | `[R,S]` | `[P,S]` | `[S,S]` |

These layouts produce two full-gradient ownership modes:

| Scenario | `full_grad` source | Ownership and lifetime |
| --- | --- | --- |
| DDP / ZeRO-1 | `[P]` view of persistent `[R]` `grad_buffer` | Created with gradient storage and released with gradient storage |
| ZeRO-2 / ZeRO-3 | Allocator-backed full `[P]` contribution | Acquired for one backward reduction and released when that reduction completes |
| HSDP | Allocator-backed full `[P,P]` contribution | Acquired for one backward reduction and released after the ordered inner/outer stages complete |

The HSDP rows show inner ZeRO-3. With outer replication, the inner axis may
instead use any 1D DDP/ZeRO row above: prepend `R` to weight, main-weight, and
gradient-storage placements and prepend `P` to gradient accumulation. Outer
optimizer sharding currently remains paired with inner ZeRO-3.

For normal FSDP, the caller's 1D mesh remains 1D; it is not expanded into a
synthetic `(1, N)` HSDP mesh:

| Value | Placement |
| --- | --- |
| Persistent model weight | `[S]` |
| Main weight / optimizer gradient | `[S]` |
| Persistent gradient storage | `[S]` |
| Microbatch gradient accumulation | `[S]` |
| Full compute weight | `[R]` |
| Local gradient contribution | `[P]` |

For HSDP with outer optimizer sharding, the layout is:

| Value | Placement |
| --- | --- |
| Persistent model weight | `[R, S]` |
| Main weight / optimizer gradient | `[S, S]` |
| Persistent gradient storage | `[R, S]` |
| Microbatch gradient accumulation | `[P, S]` |
| Full compute weight | `[R, R]` |
| Local gradient contribution | `[P, P]` |

For outer replication, main weights and optimizer gradients use `[R, S]`; the
other placements are unchanged.

## Persistent buffers

The group exposes three canonical semantic buffers internally:

```python
self.weight_buffer
self.main_weight_buffer
self.grad_buffer
```

Quantized policies may add entries to
`weight_buffers: dict[WeightBufferRole, DataParallelBuffer]`. `MODEL` is the
canonical forward representation and aliases `weight_buffer`; MXFP8 also owns a
`TRANSPOSE` representation used by backward.

`main_weight_buffer` is always the optimizer-facing representation. It may own a
distinct allocation, or it may be a placement view of `weight_buffer` when dtype
and storage permit. Optimizer DTensors are therefore always built from
`main_weight_buffer`, without an optional-main-buffer branch.

`grad_buffer` persistently owns its index, dtype, mesh, and `grad_storage`
placement, but its backing allocation is step-local. Storage is lazily bound when
backward begins. The optimizer consumes its `main_weight` placement view after
gradient finalization.

Mixed-precision formats may require another physical weight representation. Such a
representation reuses the same weight-state abstraction; it does not add HSDP
state or behavior to `DataParallelBuffer`.

## Concise runtime state

```python
class GradientPhase(Enum):
    EMPTY = auto()
    ACCUMULATING = auto()
    READY = auto()


@dataclass
class WeightRepresentationState:
    persistent: DataParallelBuffer
    valid_placements: tuple[Placement, ...]
    full: DataParallelBuffer | None = None
    pending: PendingWeightTransition | None = None


@dataclass
class GradientState:
    persistent: DataParallelBuffer
    phase: GradientPhase = GradientPhase.EMPTY
    full: DataParallelBuffer | None = None
    communication: DataParallelBuffer | None = None
```

`WeightSynchronizer` owns one `WeightRepresentationState` for each
`WeightBufferRole`. This keeps a representation's persistent buffer, valid
placement, full lease, and pending transition together rather than maintaining
parallel dictionaries. `GradientSynchronizer` independently owns
`GradientState`. Gradient placement is derived from `phase`:

| Phase | Valid gradient placement |
| --- | --- |
| `EMPTY` | none |
| `ACCUMULATING` | `layout.grad_accumulation` |
| `READY` | `layout.main_weight` |

Value existence must not be inferred from placement equality. ZeRO-2 and FSDP
use `[S]` for both accumulation and the optimizer-ready gradient.

The runtime fields record active views and allocator leases:

- `weight_state[role].full` is a full compute-weight allocation when that role's
  persistent storage cannot contain `[R, R]`;
- for DDP and ZeRO-1, `gradient_state.full` is a `[P]` view of persistent gradient
  storage and follows that storage's lifetime;
- for ZeRO-2, ZeRO-3, and HSDP, `full_grad` is an active allocator-backed
  lease and returns to `None` immediately after release;
- `gradient_state.communication` exists only when communication dtype differs
  from gradient dtype.

There is no dirty placement, cached optimizer buffer view, generic temporary-buffer
dictionary, or separate full-gradient/reduced-gradient value flags. Views are
derived from the persistent buffer and the recorded valid placements.

## Initialization

Initialization:

1. creates the three buffer layouts from the mesh and placement layout;
2. allocates and packs persistent model weights;
3. initializes or aliases optimizer main weights;
4. creates optimizer parameter DTensors from `main_weight_buffer`;
5. creates gradient-buffer metadata without allocating its backing storage;
6. records model-weight validity and an empty gradient state.

The initial state is:

```python
weight_state = {
    role: WeightRepresentationState(
        persistent=buffer,
        valid_placements=tuple(buffer.placements),
    )
    for role, buffer in weight_buffers.items()
}
gradient_state = GradientState(persistent=grad_buffer)
grad_buffer.data = None
```

For FSDP, initialized model weights are a valid `[S]` source. For HSDP, they
are a valid `[R, S]` source.

## Weight lifecycle

Unshard first restores the canonical persistent owner, then materializes full
compute weights:

```text
optimizer-valid view          persistent owner          compute weight
       [S, S]          -- outer all-gather --> [R, S]
                       -- inner all-gather --> [R, R]
```

The same transition is planned independently for each representation required by
the pass. Forward normally selects `MODEL`; MXFP8 backward selects `TRANSPOSE`.
The first transition writes into the role's persistent buffer and updates its
valid placement. The second writes into `weight_state[role].full` only when the
persistent owner is not already full. Parameters are then bound privately by
`ParameterGroup`, and the mixed-precision policy finalizes recipe-specific state.

Weight readiness is derived:

```python
all(
    weight_state[role].valid_placements == full
    or weight_state[role].full is not None
    for role in required_roles
)
```

Reshard releases every leased full representation. It does not invalidate the
persistent owners.

After an optimizer step, mixed-precision conversion copies or quantizes the
optimizer placement of `main_weight_buffer` into model-weight storage when the two
do not alias. The group then records:

```python
for state in weight_state.values():
    state.valid_placements = layout.main_weight
    state.full = None
```

The next unshard refreshes any missing persistent replicas before producing full
compute weights.

## Gradient lifecycle

When `GradientSynchronizer.ensure_storage()` allocates persistent gradient
storage for DDP or ZeRO-1, it immediately establishes `gradient_state.full` as
a `[P]` view of that storage.
The view is rebuilt whenever persistent storage migrates and invalidated when
the storage is unbound. Other strategies acquire uninitialized full-gradient
scratch at backward start. On the first microbatch, fused weight-gradient
kernels overwrite their parameter slices, ordinary gradients are copied into
their slices, and the staging layer zeroes only slices for parameters that did
not produce a gradient. On later DDP and ZeRO-1 microbatches, produced
gradients add into the persistent full-gradient value and missing gradients
leave their prior values unchanged. The whole bucket is never zeroed.

`acquire_full_grad_buffer()` is an idempotent resource-acquisition operation,
not a backward lifecycle transition. It returns the persistent full-gradient
view for DDP and ZeRO-1 or leases full-size scratch for ZeRO-2, FSDP, and HSDP.
Backward phase changes remain owned by the module and hook layer.

Every strategy follows the same two-target process:

```text
local contribution
    -> layout.grad_accumulation on every microbatch
    -> layout.main_weight on the last backward
```

For DDP and ZeRO-1, the first transition performs no collective or buffer copy:
full local gradients accumulate directly in persistent `[R]` storage viewed as
`[P]`. The last backward performs `[P] -> [R]` all-reduce for DDP or
`[P] -> [S]` reduce-scatter for ZeRO-1.

For ZeRO-2 and FSDP, every microbatch reduce-scatters `[P] -> [S]`; the final
placement is already reached, so the last transition is a no-op.

For HSDP, every microbatch performs:

```text
[P, P] -- inner reduce-scatter --> [P, S]
                                    accumulate into grad_buffer
```

On the last microbatch, its `[P, S]` output is combined with any persistent
`[P, S]` accumulation before final redistribution. The combined value may remain
in a call-local communication buffer:

```text
[P, S] -- outer reduce-scatter --> [S, S]
```

For outer replication, the final transition is an outer all-reduce:
`[P, S] -> [R, S]`.

For strategies reduced every microbatch, gradient scaling and conversion to
communication dtype happen before each redistribution. DDP and ZeRO-1 defer
both operations until the last backward so accumulated full gradients are
processed exactly once. Communication workspaces are call-local and are not
persistent parameter-group state. After every microbatch,
`release_temporary_grad_buffers()` unbinds and releases allocator-backed
full-gradient scratch, communication workspaces, and parameter bindings, then
clears the temporary `full_grad` lease. It leaves the DDP and ZeRO-1 `full_grad`
view attached to persistent gradient storage. Non-final backward sets the phase to
`ACCUMULATING`; the last backward sets it to `READY`.

`zero_grad(set_to_none=True)` resets the phase to `EMPTY`, releases temporary
gradient buffers, unbinds the DDP or ZeRO-1 `full_grad` view, detaches the local
tensors from cached optimizer-gradient DTensor wrappers, and unbinds
`grad_buffer` storage. The next backward binds a fresh `torch.empty`
allocation, rebuilds persistent views, reuses the cached DTensor wrappers, and
overwrites storage because `EMPTY` means there is no value to accumulate.
`zero_grad(set_to_none=False)` retains the allocation and its direct
full-gradient view and explicitly zeros storage to preserve its observable
zero-tensor contract.

Full-iteration CUDA graphs extend this lifetime rule. Before backward, the
group materializes `grad_buffer` and its optimizer-gradient DTensor views.
After installation, those Python objects and the persistent buffer allocation
remain stable across optimizer steps and graph replays. `zero_grad()` resets
the logical phase to `EMPTY` and zeros storage in place regardless of
`set_to_none`; it does not detach DTensor local tensors or unbind
`grad_buffer`. Full weight scratch and any strategy-required full-gradient
scratch remain transient because the CUDA graph private pool owns their replay
addresses.

## Ownership boundaries

### `DataParallelBuffer`

- owns mesh, exact placements, indexing, and bound-storage validation;
- creates placement views and placeholders;
- executes placement redistribution;
- derives the collective process group from the changed mesh axis;
- never allocates, binds parameters, or tracks training lifecycle.

### `ParameterGroup`

- owns the persistent buffers and concise validity state;
- acquires and releases full-weight scratch and, when required, full-gradient scratch;
- binds compute parameters;
- commits or accumulates gradient stages;
- performs main-to-model representation conversion;
- creates optimizer parameter and gradient DTensors.

### `FSDPModule`

- schedules streams, events, hooks, and prefetch;
- calls semantic parameter-group operations;
- does not inspect buffers, placements, or weight roles.

### `MixedPrecisionPolicy`

- owns dtype and representation conversion;
- does not own distributed storage or HSDP state.

## Required validation

Distributed tests must cover:

1. 1D FSDP `[S] -> [R]` weight materialization;
2. 1D FSDP per-microbatch `[P] -> [S]` reduction and accumulation;
3. initialized HSDP `[R, S]` weight validity;
4. first HSDP unshard `[R, S] -> [R, R]`;
5. post-optimizer unshard `[S, S] -> [R, S] -> [R, R]`;
6. non-final HSDP microbatch `[P, P] -> [P, S]` with no outer collective;
7. final-microbatch accumulation before `[P, S] -> [S, S]`;
8. outer-replicated final transition `[P, S] -> [R, S]`;
9. repeated unshard/reshard without leaked scratch leases;
10. numerical equivalence across multiple microbatches;
11. outer-to-inner weight all-gather and inner-to-outer gradient reduction
    ordering for outer-optimizer-sharded HSDP;
12. lazy gradient allocation, release on `zero_grad(set_to_none=True)`, and
    optimizer-gradient DTensor rebinding on the next step.
