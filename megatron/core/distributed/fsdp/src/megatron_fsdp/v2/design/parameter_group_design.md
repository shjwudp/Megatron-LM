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

The optimizer gradient has the same placements as `main_weight`.

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
@dataclass
class ParameterGroupState:
    weight_valid: tuple[Placement, ...]
    grad_valid: tuple[Placement, ...] | None = None
    grad_ready: bool = False
    full_weight: DataParallelBuffer | None = None
    full_grad: DataParallelBuffer | None = None
```

`weight_valid` identifies the current placement view of persistent model-weight
storage. `grad_valid` identifies the current value in persistent gradient storage.
`grad_ready` distinguishes a microbatch accumulation from an optimizer-ready
gradient when those values happen to have the same placements.

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
state.grad_valid = None
state.grad_ready = False
state.full_weight = None
state.full_grad = None
```

For FSDP, initialized model weights are a valid `[S]` source. For HSDP, they
are a valid `[R, S]` source.

## Weight lifecycle

Unshard first restores the canonical persistent owner, then materializes full
compute weights:

```text
optimizer-valid view       persistent owner       compute weight
       [S, S]          ->      [R, S]         ->     [R, R]
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

At backward start, the group acquires and zeroes `full_grad`. Autograd writes one
full local contribution, interpreted logically as all-partial placements.

For normal FSDP, every microbatch performs the complete data-parallel
reduce-scatter and accumulates into persistent optimizer placement:

```text
[P] -- per-microbatch redistribution --> [S]
                                      accumulate into grad_buffer
```

There is no separate final placement transition. `is_last_backward` only marks
the accumulated `[S]` value optimizer-ready.

For HSDP, every microbatch performs:

```text
[P, P] -- per-microbatch redistribution --> [P, S]
                                           accumulate into grad_buffer
```

The last microbatch is included in persistent `[P, S]` accumulation before final
redistribution:

```text
[P, S] -- optimizer-step redistribution --> [S, S]
```

For outer replication, the final transition is `[P, S] -> [R, S]`.

Gradient scaling and conversion to communication dtype happen once before the
per-microbatch redistribution. Communication workspaces are call-local and are
not persistent parameter-group state. After every microbatch, `full_grad` is
released. On the last backward, `grad_valid` becomes `layout.main_weight` and
`grad_ready` becomes true.

`zero_grad()` resets `grad_valid` and `grad_ready` and releases any active
`full_grad` lease.

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
10. numerical equivalence across multiple microbatches.
