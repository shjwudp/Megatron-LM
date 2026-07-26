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
- `bind()` attaches externally allocated, placement-shaped storage;
- redistribution requires an explicitly bound output when storage must grow.

The buffer should not know which `Parameter` objects consume its storage or decide how
communication results participate in gradient accumulation. Those are
`ParameterGroup` responsibilities.

This design does not add special singleton-group fast paths. A mesh dimension of size
one follows the same placement transition and collective path as any other dimension.

## Ownership

### DataParallelBuffer

The buffer owns bound-storage validation and distribution mechanics:

- `DeviceMesh`;
- logical and bound-storage placements;
- dtype and device;
- `BufferIndex`;
- binding and unbinding externally allocated tensors without freeing them;
- allocation-free placement views and aliases;
- one-axis-at-a-time redistribution;
- target-driven axis planning and coalescing for compatible buffers;
- local item and shard views.

The buffer does not import or retain a `BucketAllocator`, construct allocation keys,
allocate communication workspaces, or release temporary storage. `view()` succeeds
only when the currently bound storage contains the requested shape. A storage-growing
redistribution requires an explicitly bound `output_buffer`.

The buffer derives a process group from `mesh.get_group(mesh_dim=changed_axis)` only
when executing a redistribution. It does not cache `outer_dp_group` or
`inner_dp_group`; those names describe one current use of the mesh rather than an
intrinsic property of the buffer.

### ParameterGroup

The parameter group owns consumers and training semantics:

- the ordered parameters and parameter-to-item mapping;
- model, transpose, main-weight, and main-gradient buffer roles;
- mixed-precision policy;
- the shared temporary-storage allocator and allocation keys;
- role-indexed unsharded weight leases;
- the temporary full-gradient lease and communication-dtype workspaces;
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

Logical lifetime and physical allocation lifetime are distinct. On reshard,
`ParameterGroup` always unbinds and drops its logical lease. A storage-freeing
allocator may retain an empty tensor shell, and `TracePoolAllocator` retains its
planned physical slot and stable address, but neither case makes the parameter group
logically unsharded.

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
| `SHARD` | `Shard(0)` | The rank owns one flat shard. |
| `PARTIAL` | `Partial()` | The rank holds a contribution awaiting reduction. |

Placement describes validity, not allocation shape. A `SHARD` buffer can own compact
storage or be a slice of a `REPLICATE` buffer. The latter representation uses two
explicit objects—a replicated output placeholder and its shared-storage shard input
view—instead of encoding storage validity as a placement.

Only one mesh placement changes per `redistribute()` call. This keeps the operation
equivalent to applying one DTensor redistribution step and makes the selected mesh
dimension unambiguous. `redistribute_buffers()` is the batch planner: it applies that
primitive axis by axis, coalescing only buffers with the same process group, dtype,
device, and source placement.

`redistribute()` accepts an explicit output `DataParallelBuffer`. For an in-place
all-gather, the caller binds an externally allocated `REPLICATE` placeholder, takes
its `SHARD` view, and passes the placeholder back as the output:

```python
replicated = buffer.placeholder([Placement.REPLICATE, Placement.REPLICATE])
bucket = param_group.allocator.allocate(
    key=(param_group.param_group_id, "model_weight"),
    size=replicated.data_size,
    dtype=replicated.dtype,
    device=replicated.device,
)
replicated.bind(bucket.data)
shard = replicated.view([Placement.REPLICATE, Placement.SHARD])
shard.redistribute(replicated.placements, output_buffer=replicated)
```

The two DP-buffer objects have different placements and exact placement-shaped data,
while their tensors share one allocation.

## Core Operations

```text
FSDPModule            ParameterGroup                    DataParallelBuffer
----------            --------------                    ------------------
schedule groups  ---> select private weight buffers
                      acquire output leases
                      select final target  ------------> redistribute_buffers(target, outputs)
                                                            group compatible buffers
                                                            apply one-axis redistribute()
                      <---------------- bound outputs
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
   buffers. Replicated persistent weights are used directly; sharded weights acquire
   a role-keyed `[REPLICATE, REPLICATE]` output lease.
3. `DataParallelBuffer.redistribute_buffers()` groups compatible buffers and
   redistributes the outer placement before the inner placement into those explicit
   outputs.
4. Each owning group binds its parameters to the resulting full buffer.
5. After async communication joins the caller stream, each group finalizes its
   mixed-precision representation.
6. Resharding detaches representation-specific parameter views, unbinds each temporary
   output, and tells the allocator to release the role key.

For ZeRO-1/2, compute weights are persistently `REPLICATE`, so there is no temporary
weight lease and no allocator call. ZeRO-3 and HSDP sharded weight axes retain their
lease from asynchronous all-gather launch through the final consumer.

### Gradient reduction

1. `ParameterGroup` acquires a full-gradient lease only when persistent gradient
   storage cannot contain the replicated backward output.
2. It aliases the staged full buffer as `PARTIAL` on active DP mesh dimensions.
3. The gradient buffer performs all-reduce or reduce-scatter for the selected mesh
   axis into a caller-provided destination. `ParameterGroup` supplies a separate
   communication-dtype input when conversion or accumulation requires it.
4. `ParameterGroup.commit_comm_output()` copies or accumulates the result according to
   microbatch state.
5. The group exposes the resulting shards through optimizer-facing DTensors and
   releases the full-gradient lease after asynchronous communication completes.

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
- `storage_placements` describe the tensor currently bound to that buffer object.
- logical placements are limited to `REPLICATE`, `SHARD`, and `PARTIAL`.
- `DataParallelBuffer` has no allocator, allocation key, or temporary-buffer cache.
- `bind()` and `unbind()` never allocate or free storage.
- a buffer returned by `view()` has exact placement-shaped data and retains its
  storage owner.
- `view()` never grows storage and fails when the bound tensor cannot contain the
  requested placement.
- an explicit redistribution output shares layout, mesh, dtype, and device with its
  input and exactly matches the target placements.
- `redistribute()` changes at most one placement per call.
- `redistribute_buffers()` completes each mesh axis across compatible buffers before
  advancing to the next axis.
- parameter identity and `param_idx` are owned by `ParameterGroup`.
- only `ParameterGroup` binds storage to parameters.
- only `ParameterGroup` acquires and releases temporary storage.
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

### Phase 2: separate allocation from buffers

- Add external `bind()` / `unbind()` and unbound output placeholders.
- Move unsharded weight, full-gradient, and communication workspace leases to
  `ParameterGroup`.
- Require explicit outputs for storage-growing redistribution.
- Keep Trace Pool physical slots private to the allocator while logical leases remain
  temporary.

### Phase 3: DTensor-style placement cleanup

- Remove storage-state placements from the logical placement enum.
- Represent shared-storage redistribution with explicit DP-buffer views and an
  explicit output buffer.
- Keep logical placements limited to replicate, shard, and partial semantics.

### Phase 4: narrow role-specific policy

Move gradient communication dtype/workspace decisions and NVFP4 full-output ownership
behind `ParameterGroup`-provided transformation inputs. `DataParallelBuffer` is then a
reusable mesh-aware flat storage abstraction rather than an FSDP allocation-policy
object.

### Phase 5: immutable shared layout

Extract `BufferIndex` and other immutable layout metadata into a reusable layout
object. Bound placement buffers can then share layout without shallow-copying mutable
storage fields. This cleanup is independent of allocator decoupling.
