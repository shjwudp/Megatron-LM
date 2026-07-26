# Data-Parallel Buffer Design

## Status

This document defines the target ownership model for Megatron FSDP v2 data-parallel
buffers. The migration is incremental: existing placement transitions and collective
implementations remain valid while responsibilities move to their intended owners.

## Goals

`DataParallelBuffer` should behave like a flat, DTensor-like storage object:

- a `DeviceMesh` identifies the data-parallel topology;
- one logical placement per mesh dimension describes the current distribution;
- a `BufferIndex` maps logical tensor ranges to local flat storage;
- redistribution changes placements and returns the resulting tensor view;
- persistent and temporary storage have explicit lifetimes.

The buffer should not know which `Parameter` objects consume its storage or decide how
communication results participate in gradient accumulation. Those are
`ParameterGroup` responsibilities.

This design does not add special singleton-group fast paths. A mesh dimension of size
one follows the same placement transition and collective path as any other dimension.

## Ownership

### DataParallelBuffer

The buffer owns storage and distribution mechanics:

- `DeviceMesh`;
- current and persistent-storage placements;
- dtype and device;
- `BufferIndex`;
- persistent flat storage and temporary unsharded storage;
- allocator keys and storage release;
- one-axis-at-a-time redistribution;
- target-driven axis planning and coalescing for compatible buffers;
- local item and shard views.

The buffer derives a process group from `mesh.get_group(mesh_dim=changed_axis)` only
when executing a redistribution. It does not cache `outer_dp_group` or
`inner_dp_group`; those names describe one current use of the mesh rather than an
intrinsic property of the buffer.

### ParameterGroup

The parameter group owns consumers and training semantics:

- the ordered parameters and parameter-to-item mapping;
- model, transpose, main-weight, and main-gradient buffer roles;
- mixed-precision policy;
- selecting the weight representations required by a forward or backward pass;
- batching weight redistribution across ordered parameter groups;
- binding unsharded buffer views to parameters;
- committing or accumulating gradient communication output;
- optimizer-facing DTensor views;
- the sequence of weight, gradient, and mixed-precision transformations.

`_bind_params()` is private because it combines internal buffer identity, parameter
indexing, and mixed-precision representation rules. `unshard_model_weights()` is the
semantic entry point used by the module scheduler. `commit_comm_output()` belongs here
because copy-versus-accumulate is gradient-accumulation state, not a storage property.

### FSDPModule

The module runtime owns scheduling:

- selecting communication streams;
- prefetch, event, and hook coordination.

The module passes an ordered parameter-group sequence and lifecycle context to
`ParameterGroup.unshard_model_weights()`. It does not inspect weight-buffer roles,
choose placements, call buffer redistribution, or bind storage. Mixed-precision
finalization remains a separate semantic operation because prefetched communication
must join the caller stream before Transformer Engine kernels may launch.

### MixedPrecisionPolicy

The mixed-precision policy owns representation conversion:

- selecting storage and communication dtypes;
- exposing packed parameter data;
- binding a particular parameter representation when asked by `ParameterGroup`;
- FP8/NVFP4 quantization and post-processing.

It does not independently bind a complete buffer to a parameter group.

## Placement Model

The current implementation uses two logical placement entries ordered by mesh
dimension. Their DTensor analogues are:

| Buffer placement | DTensor analogue | Meaning |
| --- | --- | --- |
| `REPLICATE` | `Replicate()` | Every rank has a valid full logical value. |
| `FLAT` | `Shard(0)` | The rank owns a compact flat shard. |
| `PARTIAL` | `Partial()` | The rank holds a contribution awaiting reduction. |
| `DIRTY` | none | The rank-owned shard is valid inside full-sized storage. |

`DIRTY` mixes logical distribution with physical storage validity. It is retained
during the ownership migration so collective behavior does not change. The target
model represents logical sharding as `FLAT` and tracks compact-versus-full-sized
storage, plus valid ranges, separately.

Only one mesh placement changes per `redistribute()` call. This keeps the operation
equivalent to applying one DTensor redistribution step and makes the selected mesh
dimension unambiguous. `redistribute_buffers()` is the batch planner: it applies that
primitive axis by axis, coalescing only buffers with the same process group, dtype,
device, and source placement.

## Core Operations

```text
FSDPModule            ParameterGroup                    DataParallelBuffer
----------            --------------                    ------------------
schedule groups  ---> select private weight buffers
                      select final target  ------------> redistribute_buffers(target)
                                                            group compatible buffers
                                                            apply one-axis redistribute()
                      <---------------- result tensors
                      bind parameter views
wait event  --------> finalize mixed-precision views
```

`redistribute()` updates the buffer placement and returns the tensor produced by that
transition. Returning a tensor is important: communication may use a temporary dtype
workspace whose result is not yet in persistent storage. `ParameterGroup` decides how
to consume that result. Batch redistribution is therefore used only where intermediate
outputs need no parameter-group decision, such as weight all-gather or marking staged
gradients partial.

## Use Cases

### Initial packing

1. `ParameterGroup` creates buffers with a shared mesh and layout inputs.
2. It copies each policy-provided parameter representation with `set_item()`.
3. For replicated compute weights, it calls `_bind_params()` immediately.
4. It builds optimizer-facing DTensor views from the appropriate buffer.

### Weight unshard and reshard

1. `FSDPModule` schedules an ordered parameter-group sequence on the selected stream.
2. `ParameterGroup.unshard_model_weights()` privately selects the required weight
   buffers and requests the final `[REPLICATE, REPLICATE]` target.
3. `DataParallelBuffer.redistribute_buffers()` groups compatible buffers and
   redistributes the outer placement before the inner placement.
4. Each owning group binds its parameters to the resulting full buffer.
5. After async communication joins the caller stream, each group finalizes its
   mixed-precision representation.
6. Resharding returns each buffer to its persistent placements and releases temporary
   full storage.

### Gradient reduction

1. `ParameterGroup` marks staged gradients `PARTIAL` on the active DP mesh dimensions.
2. The gradient buffer performs all-reduce or reduce-scatter for the selected mesh
   axis.
3. `ParameterGroup.commit_comm_output()` copies or accumulates the result according to
   microbatch state.
4. The group exposes the resulting shards through optimizer-facing DTensors.

Gradient reduction intentionally does not use a single batched final target. Inner-DP
must reduce first, its result must be committed or accumulated, and only then may
outer-DP consume it. Keeping those one-axis transitions in `ParameterGroup` makes that
data dependency explicit.

### Main-weight update

1. The optimizer mutates the main-weight DTensor views.
2. `ParameterGroup` asks the mixed-precision policy to convert main weights into the
   compute representation.
3. When conversion requires full parameter views, such as NVFP4 master-weight
   quantization, `ParameterGroup` performs the binding before invoking the policy.
4. The policy converts values; the buffer only supplies storage and views.

### CPU offload and checkpoint views

Moving persistent storage may invalidate local DTensor views. The buffer owns the
storage move, while `ParameterGroup` rebuilds parameter and gradient views. Checkpoint
code consumes the optimizer-facing DTensors and should not reach into collective
state.

## Invariants

- `len(placements) == mesh.ndim`.
- `storage_placements` describe the persistent tensor allocation.
- `redistribute()` changes at most one placement per call.
- `redistribute_buffers()` completes each mesh axis across compatible buffers before
  advancing to the next axis.
- parameter identity and `param_idx` are owned by `ParameterGroup`.
- only `ParameterGroup` binds storage to parameters.
- only `ParameterGroup` decides whether a communication result overwrites or
  accumulates.
- `FSDPModule` does not inspect, redistribute, or bind weight buffers.
- weight-buffer selection and binding helpers remain private to `ParameterGroup`.
- process groups used by a buffer are derived from its mesh and changed axis.

## Migration

### Phase 1: ownership boundary

- Store `DeviceMesh` instead of cached outer/inner process groups.
- Move parameter binding from `DataParallelBuffer` to `ParameterGroup`.
- Move communication-result commit from `DataParallelBuffer` to `ParameterGroup`.
- Move compatible-buffer grouping and mesh-axis planning out of `FSDPModule`.
- Preserve placement transitions, collective selection, and allocation behavior.

### Phase 2: separate layout and storage

Extract immutable flat-layout metadata from mutable tensor storage. This will allow
model, transpose, main-weight, and gradient buffers to share layout where their
logical shapes permit it without sharing lifecycle state.

### Phase 3: DTensor-style placement cleanup

Replace `DIRTY` with explicit storage form and validity metadata. Keep logical
placements limited to replicate, shard, and partial semantics, and make transition
planning independent from storage allocation.

### Phase 4: narrow role-specific policy

Move gradient communication dtype/workspace decisions and quantized-weight
special-casing behind `ParameterGroup`-provided transformation inputs. At that point
`DataParallelBuffer` becomes a reusable mesh-aware flat storage abstraction rather
than an FSDP training-policy object.
