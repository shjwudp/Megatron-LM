# MFSDP v2 MXFP8: audit trail of the #6485 port onto `nvidia/main`

**Read this before using this branch for anything.**

## What this branch is

This branch is a **hand-port, performed by an AI agent, of PR #6485
(`split/pr-b-mxfp8`, head `027d20f59`) onto `nvidia/main` at
`4449eec632934d5c3d7a81e82d4257f9c46522cd`.**

It is **not** authored by the original PR author. It exists only so that #6485
and a competing MXFP8-for-MFSDP-v2 implementation can be benchmarked **on the
same base commit**. Treat its measurements as measuring *this port*, not
#6485, until the author has reviewed and blessed it.

### Why a port was necessary

#6485 is 160 commits behind current `main` and `CONFLICTING`. Those 160 commits
refactor the exact code #6485 changes, so the two cannot be text-merged:

* `main` rewrote `FsdpParameterGroup.__init__` / `_initialize_buffers` /
  `_build_fsdp_parameters` (new `_collect_parameter_metadata`,
  `DBuffer.distribute_tensors`, `DBuffer.empty(..., block_size=)`,
  `copy_parameter_attributes`, `unshard_parameters()` with no argument).
* #6485 adds `Fp8ParameterGroup(FsdpParameterGroup)`, whose overrides
  (`_init_compute_weight_storage`, `_materialize_unsharded_parameter`,
  `unshard_parameters(orientation)`) are written against the older base.
* #6485 threads a rowwise/colwise `orientation` through `module.py`'s unshard
  path; `main` has since refactored that path into
  `unshard(prefetch=...)` / `_prefetch_parameter_groups()`.

A merge of the two is therefore not a conflict resolution but a re-port of
#6485's *semantics* onto `main`'s structure. This branch has no merge commit;
it is `main` plus this port, so `git diff up/pr6485` is a real review artifact.

## What was taken verbatim from the author

These four files are byte-identical to `up/pr6485` (sha256 checked):

| file | sha256 |
| --- | --- |
| `.../experimental/quantization.py` | `83580cf016cca50cc3a54d1ccae6bc3f89fa286f5e5b331d02f713fe0800be34` |
| `.../experimental/docs/mfsdp_v2_mxfp8.md` | `915951d2daa561dfa6c442c46a6c91db5bb94f45170e6b68ad807dd86c2ec959` |
| `tests/.../mfsdp_v2/test_mxfp8_quantization.py` | `7adf46239c29b0d15b39d56e636c23cbe48228a399c1d1dfddc18f528800fa1d` |
| `tests/.../mfsdp_v2/test_mxfp8_v1_parity.py` | `4c1050a65ce389683267d1fc8ee69e105958eee20308f75bdec564cffbf43b89` |

`Fp8ParameterGroup` itself is the author's code, moved with only the mechanical
API substitutions listed below (items 5-8).

## The three structural porting items

1. **Extract `main`'s compute-weight/allocation block into the overridable hook
   `_init_compute_weight_storage()`** that `Fp8ParameterGroup` overrides, and
   extract the unsharded-parameter materialization into
   `_materialize_unsharded_parameter()`. Base bodies are `main`'s code
   verbatim; only the method boundary is new.
2. **Re-establish the rowwise/colwise `orientation` plumbing** through `main`'s
   refactored chain: `unshard(prefetch, orientation)` ->
   `_prefetch_parameter_groups(order, prefetch_size, orientation)` ->
   `_unshard_parameter_groups(orientation)` -> `group.unshard_parameters(orientation)`,
   with `pre_backward` passing `"colwise"`.
3. **Port `Fp8ParameterGroup` onto `main`'s newer `DBuffer` API**
   (`DBuffer.empty(...)`, `get_tensor_view`, `block_size`).

## Judgement calls (with the alternative rejected)

Numbering matches the review comments in the port commit.

1. **Integration method: plain commit, no merge commit.** The author's six
   commits are *not* in this branch's history. Rejected: an octopus/merge commit,
   which would make a hand-port look like a clean merge.
2. **Additive files taken verbatim** (table above) so the reviewer can diff them
   and see "unchanged".
3. **Hook extraction instead of overriding `_initialize_buffers`.** Rejected:
   overriding `_initialize_buffers` in `Fp8ParameterGroup`, which duplicates
   `main`'s `main_weight`/`main_grad` allocation and drifts further from `main`.
4. **`_init_compute_weight_storage` takes a 6th parameter, `block_size`**, which
   is not in #6485's signature. `main` computes an LCM `block_size` from all
   three placement sets and threads it into every `DBuffer` allocation; that LCM
   cannot be recomputed inside the hook because the hook does not receive the
   main-gradient placements. Rejected: omitting `block_size` (silently gives 1
   and breaks the payload/main_weight layout agreement the quantize copy-back
   depends on). `Fp8ParameterGroup` forwards `block_size` to all four payload
   `DBuffer`s for the same reason.
5. **Base `_materialize_unsharded_parameter` keeps `main`'s newer
   `copy_parameter_attributes(parameter, materialized_parameter)` call** before
   `swap_tensors`. #6485's version of that body predates that fix. Rejected:
   copying #6485's body verbatim (drops `main`'s metadata-preservation fix).
6. **Both `copy_parameter_attributes` and `sharded_parameter.__fsdp_param__ = True`
   are kept.** They do different jobs: the first copies model metadata
   (`is_embedding_or_output_parameter`, `use_muon`, ...) that `main` added; the
   second is the MFSDP-v1 marker consumed by
   `megatron/core/optimizer/clip_grads.py`. Rejected: keeping only one.
7. **`main`'s placement of the post-construction `_unsharded_model_weight`
   release is kept**, guarded with `is not None` because `Fp8ParameterGroup`
   sets it to `None`. #6485 moved that release to the end of
   `_build_fsdp_parameters`; `main`'s newer `__init__` placement wins.
8. **`reduce_scatter_stream` is dropped.** #6485 threaded it into the group so
   its base could allocate `main_grad` under
   `torch.cuda.stream(reduce_scatter_stream)`; `main` removed that scoping and
   allocates on the current stream. The parameter therefore has no consumer.
   Rejected: keeping it as an accepted-but-ignored argument (dead code).
   **This is a real behaviour difference from #6485's prototype -- see
   Known unknowns.**
9. **`fqn_to_parameter=` keyword.** `main` renamed #6485's `parameters=`
   parameter; the port uses `main`'s name.
10. **`_get_unsharded_parameter(index)` indirection is kept** (base returns
    `self.fsdp_parameters[index].unsharded`), as #6485 introduced it and uses it
    in the unshard and gradient paths.
11. **`unshard_parameters(orientation="rowwise")` is kept and plumbed through**
    `module.py`. Note this is currently a **no-op**: the only override
    (`Fp8ParameterGroup.unshard_parameters`) ignores the argument and gathers
    both orientations, documenting that "``orientation`` is accepted for
    schedule compatibility". Rejected: dropping the plumbing (loses #6485's
    declared intent and any future per-orientation optimisation).
12. **`DBuffer(...)` -> `DBuffer.empty(...)`** at the four payload allocations
    (`main` renamed the tensor-shapes-taking constructor).
13. **`get_local_tensor` -> `get_tensor_view`** at four call sites in
    `Fp8ParameterGroup` (`main`'s `DBuffer` accessor rename). This is *not* a
    global rename: `QuantizedDBuffer.get_tensor` from PR #7265 is a different
    accessor on a different class and does not exist in this tree.
14. **High-precision initialisation seeding of `main_weight` is kept.**
    #6485 seeds `main_weight` from TE's `get_high_precision_init_val()` and
    clears it before quantizing; `main` has no such handling. (PR #7114 re-adds
    the same fix independently, in a different form, which is why a competing
    branch also contains it.) The two reference-nulling statements before
    buffer allocation are kept verbatim from #6485.
15. **Base annotations relaxed to `DBuffer | None`** for `model_weight` and
    `_unsharded_model_weight`, and `_model_weight_placements` is stored, because
    `Fp8ParameterGroup` needs it to derive the payload all-gather axis with
    `changed_mesh_axis`.
16. **`assert self.model_weight is not None`** added to
    `sync_model_weight_from_main_weight` for the now-Optional annotation.
17. **`release_unsharded_storage` gets a `None` guard** in the base.
18. **`allocate_partial_grad_buffer` / `copy_gradients_to_partial_buffer` read
    gradients through `_get_unsharded_parameter(index)`**, as #6485 wrote them.
19. **`fully_shard` docstring is a union**: `main`'s `schedule_policy` /
    `register_hooks` entries plus #6485's MXFP8 note, with the note placed after
    the `Args` block rather than inside it.
20. **Adapter validation: #6485's refinement is used instead of `main`'s
    blanket rejection.** `main` rejects all FP8/FP4 for MFSDP v2; #6485 permits
    `--fp8-param-gather` with `--fp8-recipe mxfp8` and keeps rejecting
    everything else. #6485's side of that conflict *also* carried a
    `cuda_graph_impl`/`megatron_fsdp_cuda_graph_mode` rejection, which `main`
    removed during the 160 intervening commits (MFSDP v2 CUDA graphs are wired
    up in `training.py` and `param_and_grad_buffer.py`). **That rejection is NOT
    ported** -- `main` is newer there.
21. **`_group_parameters` gains `_is_fp8_parameter(parameter)` in its key** so
    MXFP8 primary weights form their own group. `main`'s
    `_specialize_placements` is deliberately untouched: #6485's design keeps
    FP8 payloads on `Flat` placements (unlike PR #7114, which introduces
    `BlockAtomic(32)` for `torch.uint8`). This relies on `MXFP8Tensor.dtype`
    being a standard floating dtype so `main`'s dtype allow-list passes;
    `_specialize_placements` is byte-identical in #6485's base and `main`, so
    #6485 made the same assumption.
22. **`Fp8ParameterGroup.__init__` still rejects symmetric memory and
    `_init_compute_weight_storage` still ignores
    `main_weight_dtype`/`main_weight_placements`/`use_symmetric_memory`**, both
    verbatim from #6485.

## Known unknowns / unresolved (author review required)

* **U1 (highest risk):** dropping `reduce_scatter_stream` (item 8) means the
  MXFP8 `main_grad` buffer is allocated on the current stream rather than the
  reduce-scatter stream. That matches `main`'s behaviour for *all* groups, but
  it differs from #6485's prototype. If the author's stream ordering mattered
  for the fp8 path, this port changes it.
* **U2:** the `orientation` plumbing (item 11) currently changes nothing,
  because the only override ignores it. If per-orientation gathering was meant
  to reduce all-gather traffic, `main`'s schedule refactor needs more than a
  pass-through.
* **U3:** `main`'s `_initialize_buffers` no longer sets
  `self._main_grad_placements`, which #6485's prototype had. It is not
  re-added; nothing in the ported code reads it (verified by grep).
* **U4:** `Fp8ParameterGroup._materialize_unsharded_parameter` keeps the
  module's existing `MXFP8Tensor` and does not run `main`'s
  `copy_parameter_attributes` on it (there is nothing to copy -- the object is
  unchanged). Flagged in case the author expected the unsharded fp8 tensor to
  carry copied metadata.
* **U5:** `test_mxfp8_v1_parity.py` documents a 4-rank run
  (`torchrun --nproc-per-node 4`) and exercises the v1 adapter path. It is not
  covered by the 2-rank MXFP8 smoke runs.
* **U6:** `main`'s `post_optimizer_model_weight` / `_model_weight_is_stale` are
  never set by `Fp8ParameterGroup._init_compute_weight_storage`. Nothing outside
  `parameter_group.py` reads them (verified by grep), but any future caller
  inherited from #6485's base would hit an unset attribute on an FP8 group.

## Verification performed on this branch

* `python -m py_compile` on every changed file.
* Symbol audit: no `get_local_tensor` remains anywhere in the tree; every
  imported name (`E4M3_BLOCK_SIZE`, `allocate_quantize_temp`, `clear_payloads`,
  `set_columnwise_payload`, `set_rowwise_payload`, `te_cast_master_weights_to_fp8`,
  `changed_mesh_axis`, `Fp8ParameterGroup`, `fp8_need_transpose_data`,
  `is_float8tensor`, `fp8_set_raw_data`, `HAVE_TE_MXFP8TENSOR`) resolves to a
  definition that exists on `main`.
* `git diff` against both `up/pr6485` (what the author wrote) and
  `nvidia/main` (what the refactor did) is the review evidence.

Cluster test results for this branch are recorded in the delivery report, not
here, because they are environment-dependent.

## Post-port fix: the port as first pushed could not run at all

Found by actually executing the branch rather than only reviewing it. The v2 arm
of the branch's own parity test raised

```
TypeError: Fp8ParameterGroup.__init__() got an unexpected keyword argument 'fqn_to_parameter'
  experiments/module.py:236
```

`main` renamed the base-class constructor kwarg `parameters` -> `fqn_to_parameter`
(`parameter_group.py:104-107` on `4449eec63`). The port updated the base class and
its call site but not the `#6485`-derived `Fp8ParameterGroup.__init__`, which had
kept its own `parameters` name and forwarded it by keyword. `up/pr6485` is
internally consistent (it predates the rename), so **this is a rebase-induced
port bug, not the author's**. Fixed in the commit that adds this note: the
subclass constructor now takes `fqn_to_parameter`. Two identifiers, no logic.

With that fix the branch's parity test runs both halves and **passes** at 2 and 4
ranks (1 node GB300, `nemo-26.08.sqsh`): max |loss rel| 2.97e-4 (2-rank) /
3.66e-4 (4-rank), max |grad_norm rel| 4.05e-4 / 4.46e-4, against tolerance
`rtol=5e-2`. That is the first numerical evidence this port has ever had.

## Prerequisite for running the parity test on this branch (NOT for training)

`tests/unit_tests/distributed/mfsdp_v2/test_mxfp8_v1_parity.py` trains twice and
uses **MFSDP v1** as its reference. Upstream commit `75e901f86` (PR #7134,
merged 2026-09-15, present on this branch's base) changed one line inside v1's
`suggested_communication_unit_size is None` branch from `max` to `min`, which
collapses v1's all-gather prefetch budget on small models and makes that v1
reference crash with `cudaErrorIllegalAddress` at the end of step 1, while
reading a parameter whose bucket storage was released.

To run the parity test, pick one:

* pass `--suggested-communication-unit-size` explicitly (any explicit value skips
  the whole inferred branch, so the `max`/`min` never applies); or
* apply the prepared upstream mitigation, which keeps #7134's inferred
  RS queue capacity but restores the all-gather floor
  (`suggested_AG_prefetch_size = max(500_000_000, value // 2)`).

**This is a prerequisite of the test, not of v2 training.** v2 is unaffected by
`75e901f86`: the inferred value is a local in v1's `MegatronFSDP.__init__`, is
never written back to `ddp_config`, and v2's adapter reads only the raw config
(`None` when the knob is unset). Benchmarking this branch's v2 path does not
require either workaround.

Note also that the mitigation is a **band-aid for the crash**, not a fix for it:
even with the floor restored, parameters remain 0-byte at the snapshot point and
read correctly only because the freed blocks have not been recycled yet. The
underlying defect is a read-after-free in v1's bucket release path.
