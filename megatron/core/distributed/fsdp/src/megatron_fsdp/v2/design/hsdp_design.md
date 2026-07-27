# HSDP Design

FSDP v2 implements HSDP as layout transitions over a two-dimensional
data-parallel mesh.

The target `ParameterGroup` ownership and runtime-state model is defined in
[`parameter_group_design.md`](parameter_group_design.md).

## Established lifecycle facts

This section is the normative starting point for the refactor. It describes outer
`optim` with inner `optim_grads_params`; later implementation work must preserve these
facts.

The placement vector is ordered as `[outer, inner]`:

| Short name | Placement |
| --- | --- |
| `R` | `REPLICATE` |
| `S` | `SHARD` |
| `P` | `PARTIAL` |

`R` and `P` have the same local extent, but not the same validity. `R` is a complete
value along that mesh dimension. `P` is an unreduced contribution. A bound `[R, S]`
buffer may therefore own the physical storage used by a current `[P, S]` logical
view; those are two `DataParallelBuffer` objects sharing one allocation.

### Persistent owners and active views

| Data | Persistent storage owner | Active or consumer view |
| --- | --- | --- |
| Model/transpose weight | `[R, S]` | Optimizer refresh source `[S, S]`; compute weight `[R, R]` |
| Main weight | `[S, S]` | Optimizer parameter `[S, S]` |
| Main gradient | `[R, S]` | Accumulating gradient `[P, S]`; optimizer gradient `[S, S]` |
| Full-gradient scratch | `[R, R]` while leased | One-microbatch contribution `[P, P]` |

The owner describes allocation capacity and the view it can contain. Only the active
view describes which values are currently valid. In particular, the main-gradient
owner remains `[R, S]`; producing an optimizer gradient does not replace it with a
compact persistent `[S, S]` buffer. The optimizer consumes an `[S, S]` view of that
owner.

### Initialization and weight materialization

1. `ParameterGroup` allocates persistent model-weight capacity as `[R, S]` and
   persistent main-weight storage as `[S, S]`.
2. Before the first optimizer update, the initialized model-weight owner is already a
   valid `[R, S]` source. Inner all-gather materializes temporary `[R, R]` compute
   weights.
3. After an optimizer update, only the optimizer-owned `[S, S]` slice is current.
   That slice is represented as an explicit view into the `[R, S]` model-weight
   owner.
4. Outer all-gather transforms `[S, S] -> [R, S]` into the containing owner.
5. Inner all-gather transforms `[R, S] -> [R, R]` into a temporary compute-weight
   output.
6. Parameters bind to `[R, R]` for computation. Reshard releases only the temporary
   `[R, R]` lease; the `[R, S]` model-weight owner persists.

There is no `DIRTY` placement or dirty flag in this model. The explicit `[S, S]`
source view records what is current, and the explicit `[R, S]` output records what
the outer all-gather will make current.

### Gradient reduction across microbatches

For every microbatch:

1. Backward stages a full local gradient contribution in `[R, R]` storage and treats
   that contribution logically as `[P, P]`.
2. Inner reduce-scatter transforms `[P, P] -> [P, S]`.
3. The result is accumulated into a persistent `[P, S]` view backed by the `[R, S]`
   main-gradient owner.

For non-final microbatches, the process stops there: there is no outer collective.
On the last backward:

4. The last inner result is first included in the accumulated `[P, S]` value.
5. Outer reduce-scatter transforms the accumulated `[P, S] -> [S, S]`.
6. The optimizer consumes that final `[S, S]` view.

Thus inner-DP reduction runs once per microbatch, while outer-DP reduction runs once
per optimizer step, after the final microbatch. The final outer output may be the
rank-owned `[S, S]` slice of the same `[R, S]` allocation used for accumulation.

For outer `no_shard`, steps 1–4 are identical. The last outer operation is an
all-reduce `[P, S] -> [R, S]`, and the optimizer consumes `[R, S]`.

### Cross-stream dependencies

Axis-specific streams do not make the two HSDP stages independent. The second
stage consumes the first stage's output, so `DataParallelBuffer.redistribute_buffers`
establishes CUDA stream dependencies in placement-transition order.

Weight materialization follows mesh order:

```text
caller stream
    -> outer AG stream: [S, S] -> [R, S]
    -> inner AG stream: [R, S] -> [R, R]
    -> bind parameters and run mixed-precision post-unshard processing
```

When the streams differ, the outer stream first waits for the caller stream and
the inner stream waits for the outer stream. If `[R, S]` is already valid, the
outer stage is skipped and the inner stream waits directly for the caller.
Parameter binding and post-unshard processing run on the last active axis stream.

#### Optional outer-DP all-gather prefetch

Outer optimizer sharding permits the persistent `[S, S] -> [R, S]` stage of a
future module to run before that module leases `[R, R]` compute-weight storage.
`outer_dp_all_gather_prefetch_depth=N` keeps at most `N` eligible future modules
in this placement stage; zero disables it. The first module's `unshard()` call
bootstraps its own outer stage and refills the configured future window after
dispatching its inner stage.

Each module has its own completion event. The current module's inner-DP stream
waits only for that module's outer event, never for the full lookahead window.
The scheduler gives the critical inner stage priority and then refills the outer
window:

```text
bootstrap:
  outer AG L0
        +--> inner AG L0
        `--> outer AG L1

steady state at L1:
  consume prefetched [R,S] L1
        +--> inner AG L1
        `--> outer AG L2
```

Depth one produces this two-stage pipeline. Larger depths can hide longer outer
latency, but are not universally faster: outer and inner collectives may contend
for the same network links. The depth is therefore a performance-tuning choice,
not a correctness requirement. When this pipeline is enabled, it replaces the
generic full-unshard-next-module prefetch so future inner all-gathers are not
launched ahead of the current module's progression.

Gradient reduction follows the inverse dependency chain:

```text
caller stream
    -> inner RS stream: [P, P] -> [P, S]
    -> accumulate the microbatch result on the inner stream
    -> outer RS/AR stream: [P, S] -> [S, S] or [R, S]
```

The final outer redistribution is invoked while the inner stream is current.
Consequently, its outer stream waits for both the inner collective and the
accumulation queued after it. `wait_stream()` supplies the CUDA event dependency;
no additional explicit event is required between the two axis collectives.

### Concrete `3 x 4` reduction example

Let the mesh shape be `(O, I) = (3, 4)`, with three outer-DP rows and four
inner-DP columns. Let `e[m, o, i]` be the full local gradient contribution
produced by microbatch `m` at rank coordinate `(o, i)`. Both partial placements
use `SUM`, and both shard placements shard the same flat gradient dimension.

At the start of a microbatch:

```text
[P, P]

[
  [e[m,0,0], e[m,0,1], e[m,0,2], e[m,0,3]],
  [e[m,1,0], e[m,1,1], e[m,1,2], e[m,1,3]],
  [e[m,2,0], e[m,2,1], e[m,2,2], e[m,2,3]],
]
```

Inner reduce-scatter independently reduces each row:

```text
[P, P] -- inner reduce-scatter --> [P, S]

r[m,o] = sum(i=0..3) e[m,o,i]
```

Each `r[m,o]` is a logical reduced tensor sharded across the four inner ranks
in row `o`; no rank in that row materializes the complete `r[m,o]`. The outer
placement remains `P`, because the three row results have not been summed.

The persistent inner-reduced accumulation after microbatch `m` is:

```text
a[o] += r[m,o]
```

It remains `[P, S]`. On the final backward, outer reduce-scatter computes:

```text
[P, S] -- outer reduce-scatter --> [S, S]

reduced_e = sum(o=0..2) a[o]
          = sum(m) sum(o=0..2) sum(i=0..3) e[m,o,i]
```

The result is fully reduced and nested-sharded along the same flat gradient
dimension: inner sharding first, then outer sharding. A `3 x 4` mesh therefore
produces 12 logical chunks, subject to the uneven-chunk rules represented by
`BufferIndex`.

## Notation

### Mesh and layout

| Symbol | Meaning |
| --- | --- |
| `O`, `I` | Outer and inner mesh sizes; the mesh shape is `(O, I)` |
| `o`, `i` | This rank's outer and inner mesh coordinates |
| `B` | Padded bucket size |
| Shard layout `(a, b)` | Outer- and inner-sharding flags; `1` is sharded and `0` is replicated |
| `unshard_dim`, `reduce_dim`, `shard_dim` | Mesh dimension ID: `0` is outer and `1` is inner |

Mesh shapes, rank coordinates, and shard layouts are distinct tuple types:
`(O, I)` is a mesh shape, `(o, i)` is a rank coordinate, and values such as
`(0, 1)` are shard layouts.

### Process groups

| Group | Role |
| --- | --- |
| `dp_cp` / `expt_dp` | Full flattened data-parallel group for dense/expert parameters |
| `intra_dp_cp` / `intra_expt_dp` | Inner-DP/EDP group when `O > 1`; otherwise the corresponding full group is used |
| `inter_dist_opt` | Outer-DP group when `O > 1`; otherwise a singleton group |

The mesh dimension names are `dp_outer` for the outer dimension and
`dp_or_edp` for the inner dense-DP or expert-EDP dimension.

### Sharding strategies

The inner `sharding_strategy` supports all four strategies below. The outer
`outer_dp_sharding_strategy` supports `no_shard` and `optim`; outer
`optim` requires inner `optim_grads_params`.

| Strategy | State sharded along the selected mesh dimension |
| --- | --- |
| `no_shard` | None |
| `optim` | Main weights and optimizer state |
| `optim_grads` | Main weights, optimizer state, and main gradients |
| `optim_grads_params` | Main weights, optimizer state, main gradients, and model/transpose weights |

See the [FSDP v2 design](design.md) for the complete strategy definitions.

## Mesh

```text
mesh dimensions: (dp_outer, dp_or_edp)
mesh shape:      (O, I)
rank coordinate: (o, i)
```

Dimension 0 is outer DP. Dimension 1 is inner DP for dense parameters and
inner EDP for expert parameters. EP remains independent.

| Parameters | Mesh | Inner group | Full flattened group |
| --- | --- | --- | --- |
| Dense | `(dp_outer, dp)` | `intra_dp_cp` or `dp_cp` | `dp_cp` |
| Expert | `(dp_outer, edp)` | `intra_expt_dp` or `expt_dp` | `expt_dp` |

For `O > 1`, dimension 0 uses `inter_dist_opt`; otherwise it is a singleton
group. Each parameter group is bound to either the dense or expert mesh.

The full group is the Cartesian product of outer and inner groups.
`mesh._flatten()` is explicitly bound to the existing `dp_cp` or
`expt_dp` process group after checking that rank order matches
`mesh.mesh.flatten()`. This is communicator binding, not data reordering.

## Layout and index

A buffer layout is:

```text
shard_layout = (outer_sharded, inner_sharded)
```

| Layout | Ownership |
| --- | --- |
| `(0, 0)` | Full bucket |
| `(0, 1)` | Inner shard |
| `(1, 0)` | Outer shard of the full bucket |
| `(1, 1)` | Outer shard of the inner shard |

Persistent storage owners and active logical views are separate buffers that may
share one allocation. In contrast, `unshard_dim`, `reduce_dim`, and `shard_dim` are
mesh dimension IDs: 0 selects outer and 1 selects inner.

Using the notation above, `BufferIndex` defines:

| Layout | Rank-owned global interval |
| --- | --- |
| `(0, 0)` | `[0, B)` |
| `(0, 1)` | `[i*B/I, (i+1)*B/I)` |
| `(1, 0)` | `[o*B/O, (o+1)*B/O)` |
| `(1, 1)` | `[i*B/I + o*B/(I*O), i*B/I + (o+1)*B/(I*O))` |

Thus:

```text
(1, 1) = shard(shard(full, inner_rank), outer_rank)
```

Both-dimension sharding is inner first, then outer. It is not the intersection
of the independent inner-only and outer-only intervals. This order matches
inner reduce-scatter followed by outer reduce-scatter.

Each buffer owns a `BufferIndex` built from the same parameter order and
alignment. It contains:

- `ItemIndex`: item offset, size, and shape in the full bucket.
- `ShardMeta`: this rank's interval for each shard layout.

For an item interval `P` and layout interval `S`:

```text
self range  = intersection(P, S) - P.start
local range = intersection(P, S) - S.start
```

This mapping lets different buffer roles address the same parameter while
using different physical layouts.

DTensor placements are indexed by mesh dimension, while the argument to
`Shard` is a logical tensor dimension. `[Shard(0), Shard(0)]` means that
both outer and inner mesh dimensions shard tensor dimension 0.
`BufferIndex` selects the physical `(1, 1)` local slice before it is wrapped
as a DTensor. `_shard_order = [1, 0]` is used only when computing uneven-
DTensor checkpoint chunk metadata, so global offsets follow the same
inner-then-outer layout. It does not reorder the mesh or control collectives.

## Buffer layouts and conversion

For either mesh dimension, the strategy-to-storage rule is:

| Buffer role | `no_shard` | `optim` | `optim_grads` | `optim_grads_params` |
| --- | ---: | ---: | ---: | ---: |
| Model/transpose weight | 0 | 0 | 0 | 1 |
| Main weight | 0 | 1 | 1 | 1 |
| Main gradient | 0 | 0 | 1 | 1 |

Outer and inner decisions are combined as
`(outer_sharded, inner_sharded)`. For outer `optim` plus inner
`optim_grads_params`:

| Data | Persistent placement | Active view |
| --- | --- | --- |
| Model/transpose weight | `[R, S]` | Optimizer refresh source `[S, S]`; compute weight `[R, R]` |
| Main weight | `[S, S]` | Optimizer parameter `[S, S]` |
| Main gradient | `[R, S]` | Accumulation `[P, S]`; optimizer gradient `[S, S]` |

`tensor_view()` and `copy_tensors_()` intersect the requested tensor range with the
range present in persistent storage, then translate it to storage-local
coordinates.

Bound buffer views follow three rules:

1. `view(layout)` returns storage for an exact layout.
2. It returns a view when bound storage contains the requested shard, such as
   `(0, 1) -> (1, 1)`.
3. Otherwise `ParameterGroup` acquires an external destination, creates a
   placeholder for it, and binds the allocation before redistribution, such as
   `(1, 1) -> (0, 1)`.

An all-gather changes one layout bit from 1 to 0. A reduce-scatter changes one
bit from 0 to 1. An all-reduce leaves the layout unchanged. Parameters are
bound only from `(0, 0)`; reshard unbinds group-owned leases before returning
their keys to the allocator.

## State transitions

The flows below assume inner `optim_grads_params`. The optimizer-step
boundary is identified by `set_is_last_backward(True)`. The final backward
callback marks the model-weight refresh pending at that boundary. An integrated
optimizer may install updated model weights immediately; otherwise the next
normal root pre-forward installs them before any outer or inner unshard.
Activation-recompute forwards do not consume the pending refresh.

The unshard tables distinguish the persistent model-storage owner from the
currently valid weight view.

### Outer `no_shard`

#### Reduce grad

| Step | Frequency | Operation | Active placement |
| --- | --- | --- | --- |
| Backward contribution | Every microbatch | — | `[P, P]` |
| Reduce inner contribution | Every microbatch | Inner reduce-scatter | `[P, S]` |
| Accumulate inner result | Every microbatch | Local accumulation | `[P, S]` |
| Synchronize outer replicas | Last backward only | Outer all-reduce | `[R, S]` |
| Optimizer consumes gradient | Step boundary | — | `[R, S]` |

The outer all-reduce gives every outer rank the same inner gradient shard.
Main weights and optimizer state are also replicated on outer, so every outer
rank applies the same update.

#### Unshard

| Step | Operation | Persistent model owner | Current valid weight view |
| --- | --- | --- | --- |
| Main weight, replicated on outer | — | — | `[R, S]` |
| Copy the complete inner shard into model storage | Copy | `[R, S]` | `[R, S]` |
| Skip outer all-gather | — | `[R, S]` | `[R, S]` |
| Materialize compute weight | Inner all-gather | `[R, S]` | `[R, R]` |
| Bind parameters | — | `[R, S]` | `[R, R]` |

The model-weight storage is already complete on outer after the copy. Reshard
releases the temporary `[R, R]` compute buffer and keeps the persistent
`[R, S]` owner.

### Outer `optim`

#### Reduce grad

| Step | Frequency | Operation | Active placement |
| --- | --- | --- | --- |
| Backward contribution | Every microbatch | — | `[P, P]` |
| Reduce inner contribution | Every microbatch | Inner reduce-scatter | `[P, S]` |
| Accumulate inner result in `[R, S]` owner | Every microbatch | Local accumulation | `[P, S]` |
| Shard across outer ranks | Last backward only | Outer reduce-scatter | `[S, S]` |
| Optimizer consumes gradient | Step boundary | — | `[S, S]` |

The outer reduce-scatter leaves each outer rank with one slice of the inner
gradient shard. Main weights and optimizer state use the same `[S, S]`
ownership. The main-gradient allocation remains owned by the persistent
`[R, S]` buffer; `[S, S]` is its optimizer-facing final view.

#### Unshard

| Step | Operation | Persistent model owner | Current valid weight view |
| --- | --- | --- | --- |
| Main weight | — | — | `[S, S]` |
| Copy the local slice into model storage | Copy | `[R, S]` | `[S, S]` |
| Reconstruct the complete inner shard | Outer all-gather | `[R, S]` | `[R, S]` |
| Materialize compute weight | Inner all-gather | `[R, S]` | `[R, R]` |
| Bind parameters | — | `[R, S]` | `[R, R]` |

Immediately after the optimizer-to-model copy, only the explicit `[S, S]`
source view is current. Outer all-gather writes into the containing `[R, S]`
owner, making that output current. No dirty state is stored on
`DataParallelBuffer`; the source and output objects make the transition
explicit. Inner unshard then writes temporary `[R, R]` compute weights.
