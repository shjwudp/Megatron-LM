# Parameter Group Design

## Status

This document defines the target ownership and state model for Megatron FSDP v2
`ParameterGroup`. The implementation may migrate to this model incrementally, but
each intermediate step must preserve the HSDP lifecycle in
[`hsdp_design.md`](hsdp_design.md).

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

The group has three persistent distributed values:

- model weights used to materialize compute parameters;
- optimizer main weights;
- persistent accumulated or reduced gradients.

It may temporarily lease a full model-weight buffer and a full gradient buffer.
Those leases are logical runtime state, not persistent buffer roles.

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

The group exposes three semantic buffers internally:

```python
self.weight_buffer
self.main_weight_buffer
self.grad_buffer
```

`main_weight_buffer` is always the optimizer-facing representation. It may own a
distinct allocation, or it may be a placement view of `weight_buffer` when dtype
and storage permit. Optimizer DTensors are therefore always built from
`main_weight_buffer`, without an optional-main-buffer branch.

`grad_buffer` owns enough persistent storage for `grad_storage`. The optimizer
consumes its `main_weight` placement view after gradient finalization.

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
class ParameterGroupState:
    weight_valid: tuple[Placement, ...]
    grad_phase: GradientPhase = GradientPhase.EMPTY
    full_weight: DataParallelBuffer | None = None
    full_grad: DataParallelBuffer | None = None
```

`weight_valid` identifies the current placement view of persistent model-weight
storage. Gradient placement is derived from `grad_phase`:

| Phase | Valid gradient placement |
| --- | --- |
| `EMPTY` | none |
| `ACCUMULATING` | `layout.grad_accumulation` |
| `READY` | `layout.main_weight` |

Value existence must not be inferred from placement equality. ZeRO-2 and FSDP
use `[S]` for both accumulation and the optimizer-ready gradient.

The two buffer fields are optional scratch leases:

- `full_weight` is a full compute-weight allocation when persistent weight storage
  cannot contain `[R, R]`;
- `full_grad` contains the current backward contribution.

There is no dirty placement, cached optimizer buffer view, generic temporary-buffer
dictionary, or separate full-gradient/reduced-gradient value flags. Views are
derived from the persistent buffer and the recorded valid placements.

## Initialization

Initialization:

1. creates the three buffer layouts from the mesh and placement layout;
2. allocates and packs persistent model weights;
3. initializes or aliases optimizer main weights;
4. creates optimizer parameter DTensors from `main_weight_buffer`;
5. creates persistent gradient storage;
6. records model-weight validity and an empty gradient state.

The initial state is:

```python
state.weight_valid = weight_buffer.placements
state.grad_phase = GradientPhase.EMPTY
state.full_weight = None
state.full_grad = None
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

The first transition writes into `weight_buffer` and updates `weight_valid`. The
second transition writes into `full_weight` only when the persistent owner is not
already full. Parameters are then bound privately by `ParameterGroup`.

Weight readiness is derived:

```python
weight_valid == full or full_weight is not None
```

Reshard unbinds parameter representations and releases `full_weight`. It does not
invalidate the persistent owner.

After an optimizer step, mixed-precision conversion copies or quantizes the
optimizer placement of `main_weight_buffer` into model-weight storage when the two
do not alias. The group then records:

```python
state.weight_valid = layout.main_weight
state.full_weight = None
```

The next unshard refreshes any missing persistent replicas before producing full
compute weights.

## Gradient lifecycle

At backward start, the group acquires uninitialized `full_grad` storage. Fused
weight-gradient kernels overwrite their parameter slices, ordinary gradients are
copied into their slices, and the staging layer zeroes only slices for parameters
that did not produce a gradient. The whole bucket is never zeroed.

Every strategy follows the same two-target process:

```text
local contribution
    -> layout.grad_accumulation on every microbatch
    -> layout.main_weight on the last backward
```

For DDP and ZeRO-1, the first transition performs no collective: full local
gradients accumulate as `[P]`. The last backward performs `[P] -> [R]`
all-reduce for DDP or `[P] -> [S]` reduce-scatter for ZeRO-1.

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

Gradient scaling and conversion to communication dtype happen once before the
per-microbatch redistribution. Communication workspaces are call-local and are
not persistent parameter-group state. After every microbatch, `full_grad` is
released. Non-final backward sets the phase to `ACCUMULATING`; the last backward
sets it to `READY`.

`zero_grad(set_to_none=True)` resets the phase to `EMPTY`, releases any active
`full_grad` lease, and detaches optimizer-facing gradients without clearing
`grad_buffer`. The next reduction overwrites stale storage because `EMPTY` means
there is no value to accumulate. `zero_grad(set_to_none=False)` retains an
explicit buffer zero to preserve its observable zero-tensor contract.

## Ownership boundaries

### `DataParallelBuffer`

- owns mesh, exact placements, indexing, and bound-storage validation;
- creates placement views and placeholders;
- executes placement redistribution;
- derives the collective process group from the changed mesh axis;
- never allocates, binds parameters, or tracks training lifecycle.

### `ParameterGroup`

- owns the persistent buffers and concise validity state;
- acquires and releases full weight/full gradient scratch;
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
    ordering for outer-optimizer-sharded HSDP.
