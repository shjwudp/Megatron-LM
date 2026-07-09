# Compile-Native Megatron-FSDP (M-FSDP Compile) — Design Proposal

## 1. Summary

Add a **torch.compile-native** FSDP implementation alongside M-FSDP v2, inspired by
[SimpleFSDP](https://github.com/pytorch/torchtitan). Sharding/unsharding is expressed as
**plain differentiable tensor ops** (functional collectives), so the whole train step is a
single traceable graph and the compiler (AOTAutograd + Inductor) owns bucketing, comm/compute
overlap, and memory — instead of eager hooks, manual CUDA streams, custom autograd `Function`s,
`TracePoolAllocator`, and the `te_graph_runtime` CUDA-graph path.

It lives in a **separate subpackage** and shares the v2 public interface (`fully_shard`,
`FSDPModule`, `MixedPrecisionPolicy`), selectable by an option. v2 is unchanged and remains the
default.

## 2. Motivation

- The v2 eager machinery (per-module `unshard`/`reshard` hooks, `.data` rebinding, storage
  `resize_(0)`, weak refs, side-streams, `Graphed.apply`, trace-pool slot planning) is
  fundamentally **graph-break-hostile**. Making v2 itself compile-clean would be an invasive
  rewrite of its hot path.
- A compile-native path lets us **delete hand-written overlap/bucketing/CUDA-graph code** and
  inherit the compiler's passes, which is where the ecosystem (torchtitan/SimpleFSDP) is
  investing.
- Clean separation lets us ship it experimentally, compare against v2 on the same models/tests,
  and converge later.

## 3. Background: why the two differ

| Concern | M-FSDP v2 (eager) | Compile-native (this proposal) |
|---|---|---|
| Unshard | `unshard()` hook, all-gather into a pooled buffer, `_replace_module_parameter` rebinds `param.data` | differentiable all-gather op inside the traced graph; forward returns the full weight |
| Reshard | `reshard()` hook frees buffer | recompute unshard in backward via activation checkpoint (no persistent full weight) |
| Grad reduce | `reduce_grad()` hook -> `main_grad` -> reduce-scatter on side stream | reduce-scatter is the **backward** of the unshard op, emitted into the graph |
| Overlap/bucketing | manual streams + prefetch + `TracePoolAllocator` | Inductor comm passes (bucketing, reordering) |
| CUDA graph | `te_graph_runtime` capture + stable-address plumbing | `mode="reduce-overhead"` / compiler-managed cudagraphs |
| Param object | swapped between `dist_param`/`param` | single sharded `DTensor`, unshard is functional |

The v2 grad-ownership and trace-pool bugs we've been fixing are all symptoms of the eager
approach; the compile path sidesteps them by construction.

## 4. Goals / Non-goals

**Goals**
- Correct ZeRO-3 (`optim_grads_params`) FSDP under `torch.compile(fullgraph=True)` for a
  transformer block stack.
- Shared public API with v2; opt-in.
- HSDP (2D `dp_outer x dp`) and mixed precision (BF16/FP8) as fast-follows.
- Numerical parity with v2 on the toy + QwenImage convergence tests we already have
  (`--real-data`).

**Non-goals (initially)**
- CUDA-graph-specific code (delegated to the compiler).
- NVFP4 primary weights, CPU offload, uneven-shard checkpoint parity — deferred.
- Replacing v2. This is additive and experimental.

## 5. Directory layout & shared interface

```
megatron_fsdp/v2/
|-- fully_shard.py          # v2 (unchanged) - dispatches on backend
|-- compile/                # NEW subpackage
|   |-- __init__.py         # fully_shard, FSDPModule re-exports
|   |-- fully_shard.py      # entrypoint: parametrize + register groups
|   |-- parametrize.py      # module parametrization / weight provider
|   |-- collectives.py      # differentiable all-gather / reduce-scatter
|   |-- param_group.py      # sharded DTensor param groups (reuses buffer_index)
|   |-- mixed_precision.py  # thin adapter over v2 MixedPrecisionPolicy
|   `-- compile_config.py   # Inductor pass toggles (bucketing/overlap)
`-- design/compile_fsdp_design.md
```

Enablement — one selector, existing signature preserved:

```python
fully_shard(module, mesh=mesh, mp_policy=mp, sharding_strategy="optim_grads_params",
            backend="compile")   # default "eager" (== current v2)
```

`backend="compile"` dispatches to `v2.compile.fully_shard`. Reuse v2 types
(`MixedPrecisionPolicy`, sharding-strategy strings, `BufferIndex` for layout/uneven-shard
metadata) so checkpointing and configs stay compatible.

## 6. Core design

**6.1 Sharded parameter representation.** Each parameter becomes a `DTensor` with `[Shard(0)]`
(dense) or `[Shard(0), Shard(0)]` (HSDP) over the DP mesh — reuse `BufferIndex`/`uneven_dtensor`
for padded, even-sharded layout so checkpoints match v2.

**6.2 Differentiable unshard (the crux).** A single autograd-aware op whose forward all-gathers
and backward reduce-scatters, built on **functional collectives**
(`torch.ops._c10d_functional.all_gather_into_tensor` / `reduce_scatter_tensor`) so it traces
cleanly:

```python
class _Unshard(torch.autograd.Function):  # or a functional/custom-op equivalent
    @staticmethod
    def forward(ctx, sharded, group, mp_dtype):
        ctx.group, ctx.numel = group, sharded.numel()
        return all_gather_into_tensor(sharded.to(mp_dtype), group)   # full weight
    @staticmethod
    def backward(ctx, grad_full):
        return reduce_scatter_tensor(grad_full, ctx.group), None, None
```

(We'll evaluate `DTensor.redistribute(Replicate())` vs. an explicit custom op — the custom op
gives us more control over dtype/casting and is what SimpleFSDP-style stacks lean on. Decision
in Phase 1.)

**6.3 Module parametrization.** Use `nn.utils.parametrize` (or a lightweight forward-pre
wrapper) so `module.weight` yields `_Unshard(sharded_weight, group, param_dtype)` during
forward. No `.data` swapping, no hooks.

**6.4 Reshard-after-forward = activation checkpointing.** Wrap the unshard in AC/SAC so the full
weight is freed after forward and **recomputed** in backward. This is how the compile path gets
FSDP's memory savings without manual free.

**6.5 Mixed precision.** Cast inside the unshard op (param -> compute dtype); grads reduce in
`grad_comm_dtype`. Reuse `MixedPrecisionPolicy` fields; FP8/MXFP8 param-gather as a Phase 4 item
(cast before all-gather).

**6.6 HSDP.** Two nested unshards (inner all-gather, then outer all-gather / outer all-reduce for
`optim`), mirroring v2's `(outer, inner)` layout and `hsdp_design.md`. Backward emits
inner-then-outer reduce-scatter to match the documented ordering.

**6.7 Compiler passes.** Expose Inductor knobs (comm bucketing, comm-compute reordering,
optional `mode="reduce-overhead"` cudagraphs) in `compile_config.py`, defaulting conservative.
This replaces the manual prefetch/bucket/stream logic. See Section 12 for the concrete control
surface.

## 7. Feature-parity matrix (target)

| Feature | Phase | Notes |
|---|---|---|
| `optim_grads_params` (ZeRO-3), dense DP | 1 | MVP |
| Mixed precision BF16 | 1 | |
| `no_shard` / `optim` / `optim_grads` | 2 | |
| HSDP 2D | 3 | reuse hsdp layout |
| FP8/MXFP8 param gather | 4 | |
| DCP checkpoint parity with v2 | 2-3 | via shared `BufferIndex`/`uneven_dtensor` |
| CPU offload, NVFP4 | later | |

## 8. Limitations / constraints (call out up front)

- Requires `torch.compile(fullgraph=True)`-clean model code (no unguarded graph breaks in the
  wrapped blocks).
- No eager hooks/side-streams; no `te_graph_runtime`, no `TracePoolAllocator`.
- Custom-op/DTensor collective support must match the target torch version (pin & test).
- Optimizer must consume sharded `DTensor` grads (works with the existing dist-optimizer path,
  to be validated).

## 9. Phased implementation plan

1. **Phase 0 — scaffolding & interface.** `compile/` package, `backend=` dispatch in
   `fully_shard`, `FSDPModule` shim, config object, no behavior yet. Land with a skipped/xfail
   test.
2. **Phase 1 — MVP ZeRO-3 dense.** Sharded `DTensor` params + `_Unshard` op + parametrization +
   AC-based reshard, BF16. Single-block then N-block toy under `torch.compile(fullgraph=True)`.
   **Gate: `examples/megatron_fsdp/fsdp_toy.py --use-real-data` converges and matches eager
   within tolerance.**
3. **Phase 2 — strategies + checkpoint.** Add `no_shard/optim/optim_grads`; DCP save/load parity
   with v2 via shared layout metadata.
4. **Phase 3 — HSDP.** 2D mesh, nested unshard, ordering per `hsdp_design.md`. Gate on
   `test_qwenimage.py --real-data --enable-hsdp`.
5. **Phase 4 — perf + FP8.** Turn on Inductor comm bucketing/overlap, benchmark vs v2 (step time
   + peak mem); add FP8 param gather.
6. **Phase 5 — hardening.** mcore adapter wiring, docs, broader unit tests, decide default-off ->
   opt-in-recommended.

Each phase = one reviewable PR with its own tests.

## 10. Testing strategy

- Reuse the `--real-data` convergence harnesses (teacher-student toy, flow-matching QwenImage) as
  the correctness oracle — same script, `backend="compile"`.
- New unit tests under `tests/unit_tests/distributed/megatron_fsdp/v2/compile/`: single-op
  all-gather/reduce-scatter grad correctness, parametrized-weight numerics vs. a dense reference,
  `fullgraph=True` compiles with zero graph breaks (assert via `torch._dynamo.explain`).
- Parity gate: compile vs eager final-loss ratio within tolerance on identical seeds/data.

## 11. Open questions (to resolve in Phase 1)

1. `DTensor.redistribute` vs. explicit functional-collective custom op for unshard — which
   composes better with Inductor bucketing on our torch pin?
2. Reshard via AC vs. keeping full weights when memory allows — expose as a policy knob?
3. How much of v2's `param_group`/`BufferIndex` can be reused verbatim vs. a slimmer
   sharded-DTensor group?
4. Interaction with pipeline `combined_1f1b` overlap — does the compiled block coexist, or is PP
   out of scope initially?
5. Minimum supported torch version for the functional-collective + compile feature set.

## 12. Controlling / customizing the compiler

The compile backend is only useful if we can steer *what* the compiler does. There are four
layers of control, from coarse to fine; `compile_config.py` centralizes them behind a
`CompileConfig` dataclass so callers get one knob and we keep the torch-version-specific bits in
one place.

### 12.1 Where we invoke the compiler

We do **not** ask users to `torch.compile` the model themselves. `fully_shard(..., backend="compile")`
compiles the wrapped units (per transformer block, matching v2's `fsdp_unit_modules`) so the
unshard collectives are inside each compiled region:

```python
compiled = torch.compile(
    block,
    backend=cfg.backend,          # "inductor" (default) or a custom backend
    mode=cfg.mode,                # None | "default" | "reduce-overhead" | "max-autotune"
    fullgraph=cfg.fullgraph,      # True: fail loudly on graph breaks (recommended for FSDP)
    dynamic=cfg.dynamic,          # False for static shapes (typical training)
    options=cfg.inductor_options, # dict passed straight to Inductor (see 12.3)
)
```

`CompileConfig` (sketch):

```python
@dataclass
class CompileConfig:
    backend: str = "inductor"
    mode: Optional[str] = None
    fullgraph: bool = True
    dynamic: bool = False
    # comm optimization
    enable_comm_bucketing: bool = True
    enable_comm_reordering: bool = True     # comm/compute overlap
    bucket_size_mb: Optional[float] = None
    # escape hatches
    inductor_options: dict = field(default_factory=dict)
    inductor_config_patches: dict = field(default_factory=dict)  # torch._inductor.config keys
    dynamo_config_patches: dict = field(default_factory=dict)    # torch._dynamo.config keys
```

### 12.2 Dynamo-level control (tracing)

- **`fullgraph=True`** is the default for FSDP: a graph break would split the unshard collective
  from the compute it must overlap with, silently killing performance. We assert zero breaks in
  tests via `torch._dynamo.explain(...)`.
- **Guards / recompiles:** `dynamic=False` pins static shapes (avoids recompiles); we surface
  `torch._dynamo.config` patches (e.g. `cache_size_limit`, `recompile_limit`) through
  `dynamo_config_patches` for debugging recompilation storms.
- **Selective disable:** for code that legitimately cannot be traced (e.g. a custom loss branch),
  `@torch._dynamo.disable` / `torch.compiler.disable` on that callable — same pattern v2 already
  uses on its hooks (`@torch.compiler.disable`).

### 12.3 Inductor-level control (codegen + comm passes)

This is where FSDP's overlap actually happens. The relevant knobs (names may shift across torch
versions — hence the `inductor_config_patches` escape hatch):

- **Comm/compute reordering (overlap):** `torch._inductor.config.reorder_for_compute_comm_overlap`
  and its pass list `reorder_for_compute_comm_overlap_passes`. This replaces v2's manual
  prefetch/side-stream scheduling.
- **Collective bucketing:** the all-gather/reduce-scatter bucketing passes (e.g.
  `bucket_all_gathers_fx` / `bucket_reduce_scatters_fx`, or the newer "auto bucketing" config on
  recent torch). This replaces v2's `TracePoolAllocator` bucket sizing. `bucket_size_mb` maps to
  the pass's size threshold.
- **CUDA graphs:** `mode="reduce-overhead"` (or `triton.cudagraphs`) lets Inductor own graph
  capture — replacing `te_graph_runtime` entirely. We keep this **off by default** until the
  functional-collective path is proven, then enable behind the flag.
- **Autotuning:** `mode="max-autotune"` / `max_autotune_gemm` for kernel selection; off by default
  (compile-time cost), opt-in for benchmarking.

We apply these via a context manager so they are scoped to capture and never leak into the rest
of the process:

```python
@contextlib.contextmanager
def _apply_inductor_patches(cfg):
    import torch._inductor.config as ind
    saved = {k: getattr(ind, k) for k in cfg.inductor_config_patches}
    try:
        for k, v in cfg.inductor_config_patches.items():
            setattr(ind, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(ind, k, v)
```

### 12.4 Custom backends / passes (deepest control)

When the built-in passes are insufficient we have two escape hatches, both standard torch APIs:

- **Custom Inductor FX passes:** register `post_grad_custom_post_pass` /
  `post_grad_custom_pre_pass` (or `joint_custom_*`) to run our own graph transform — e.g. an
  M-FSDP-aware bucketing heuristic that groups all-gathers by parameter group instead of by size.
- **Custom compile backend:** pass a callable `backend=my_backend(gm, example_inputs)` to
  `torch.compile`. This gives full control of the AOTAutograd graph (we can splice collectives,
  choose overlap points, or fall back to `inductor` after our pass). Use only if the FX-pass
  hooks are not enough.

### 12.5 Observability / debugging knobs

- `torch._dynamo.explain(fn)(*inputs)` — assert graph-break count == 0 in unit tests.
- `TORCH_LOGS="graph_breaks,recompiles,inductor"` / `torch._logging.set_logs(...)` — surfaced via
  a `CompileConfig.debug_logs` convenience.
- `torch._inductor.config.trace.enabled` + the generated code dump to inspect whether collectives
  were bucketed/reordered as intended.
- A parity/regression gate that compares the compiled step's collective count and ordering
  against expectations (guards against a torch upgrade silently disabling a pass).

### 12.6 Versioning discipline

Inductor comm-pass config names change between torch releases. Rule: **never reference a
`torch._inductor.config` attribute directly in hot code** — go through `compile_config.py`, which
feature-detects (`hasattr`) and maps our stable `CompileConfig` fields onto whatever the pinned
torch exposes, raising a clear error if a required pass is unavailable. This keeps base-image
bumps (see `mcore-bump-base-image`) from breaking the backend silently.
