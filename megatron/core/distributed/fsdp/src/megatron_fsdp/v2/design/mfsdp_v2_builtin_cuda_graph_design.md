# Megatron FSDP v2 built-in CUDA graph design

> Experimental: this document describes the per-FSDP-module CUDA graph path
> built into Megatron FSDP v2. Full-iteration CUDA graph capture is documented
> separately in [`full_iteration_cuda_graph_design.md`](full_iteration_cuda_graph_design.md).

## Scope

Megatron FSDP v2 can capture selected leaf FSDP modules with CUDA graphs while
keeping FSDP unshard, reshard, and gradient-reduction work outside the graphed
region.

This path is enabled per FSDP module with `enable_cuda_graph=True`. It is useful
when only selected modules are graph-safe or when full-iteration capture is not
desired.

## How to enable it

Direct `fully_shard()` usage:

```python
for layer in model.layers:
    fully_shard(layer, enable_cuda_graph=True)

fully_shard(model)  # root wrapper; do not enable CUDA graph on this parent
```

Megatron training CLI usage for Megatron FSDP v2:

```bash
--use-megatron-fsdp-v2 \
--mfsdp-cuda-graph attn mlp
```

Supported `--mfsdp-cuda-graph` module selectors are:

- `transformer` for `TransformerLayer`
- `mamba` for `MambaLayer`
- `attn` for attention modules
- `mlp` for dense MLP modules
- `moe` for MoE expert MLP modules
- `moe_router` for MoE router modules

Only non-nested leaf FSDP modules are eligible. A parent FSDP module that
contains other FSDP modules cannot also use `enable_cuda_graph=True`; the
runtime raises a `RuntimeError` for that configuration.

## Why FSDP needs special handling

CUDA graph replay uses the same memory addresses that were observed during
capture. FSDP normally materializes full parameters only temporarily:

```text
forward pre-hook  -> unshard parameters into a temporary full buffer
forward compute   -> read full parameters
forward hook      -> reshard and release the temporary full buffer
```

A normal temporary allocator may return a different address on the next
microbatch. That is not compatible with CUDA graph replay.

Megatron FSDP v2 solves this with two components:

1. [`TracePoolAllocator`](../allocator.py) traces one microbatch, builds a
   static key-to-slot plan, and gives each planned FSDP buffer a stable address.
2. `te_graph_runtime.make_graphed_callables()` supports `capture_time_hooks`,
   so FSDP can run unshard/reshard during warmup and capture without recording
   those hooks into the CUDA graph.

The graph captures module math. FSDP memory movement and gradient reduction
remain in normal Python hooks around the graph.

## Lifecycle

```text
Microbatch 0: trace
  eager forward/backward
  TracePoolAllocator records alloc/free events
  post-backward final callback calls plan()
  allocator phase becomes "optimized"

Microbatch 1: record
  forward hooks unshard selected modules into stable trace-pool slots
  CudaGraphRunner records sample inputs and outputs
  backward runs eagerly
  post-backward final callback requests capture

Microbatch 2 root forward pre-hook: capture
  waits until the previous training schedule has left backward
  calls capture_and_install() before the next parameter all-gather
  selected module forward methods are replaced with graphed callables

Microbatch 2+: replay
  FSDP hooks still run around each module call
  unshard places parameters back into the same trace-pool slots
  module forward/backward replay CUDA graphs
  post-backward hooks reshard and reduce gradients outside the graph
```

## Activation recompute

`cuda_graph_activation_recompute=True` captures three programs for each
selected module:

- `F`: the original forward program, whose module kernels are captured under
  `no_grad` so their intermediate activations are not retained;
- `RF`: the grad-enabled forward run by checkpoint recomputation;
- `B`: backward, which consumes the autograd state produced by `RF`.

The checkpoint scope may be larger than the CUDA Graph scope. For example,
`checkpoint(A -> CG(B) -> C)` graphs only `B`. No checkpoint wrapper or phase
marker is required. The M-FSDP pre-forward hook selects `F` during the normal
forward phase and `RF` when the same module runs from an active autograd graph
task during M-FSDP backward.

Automatic phase discovery currently requires non-reentrant checkpointing. Its
original forward is grad-enabled, while recomputation runs during M-FSDP
backward. A reentrant original forward runs under `no_grad` and cannot be
distinguished from a train-mode evaluation or probe without an explicit
checkpoint marker.

Every grad-disabled call is inference. It neither creates pending backward
state nor changes an existing training invocation. A grad-enabled forward
during backward is accepted as `RF` only when the selected invocation awaits
recompute and the checkpoint runtime identifies an active autograd graph task.
This also identifies recomputation from Transformer Engine checkpointing,
which does not set the MCore checkpoint flag.
Other side forwards are rejected before static inputs are overwritten. A
grad-enabled call to the same module from inside that active task cannot be
distinguished from its real recomputation and is unsupported.
An original forward whose outputs are all detached is rejected and its pending
state is released because no output can start the matching backward.

```text
normal forward:
  all-gather B parameters -> replay F_B -> reshard

checkpoint recompute:
  all-gather B parameters -> replay RF_B -> reshard

backward:
  all-gather B parameters -> replay B_B -> reduce-scatter gradients -> reshard
```

Parameter communication, resharding, and gradient reduction stay outside the
graphs. Only address-stable module computation is captured.

Capture is deferred from the backward final callback to the next root forward
pre-hook. This avoids synchronizing CUDA while a pipeline stage may still have
point-to-point communication in flight. Evaluation and grad-disabled forwards
leave the request pending until the next grad-enabled training forward.

The runner records the observed `F`, `RF`, and `B` order before capture and
infers serial checkpoint regions from that trace. A region runs `RF` in forward
module order and `B` in reverse module order. With multiple pending forwards,
the runner encodes the observed region and microbatch schedule in the runtime
`_order` argument and assigns one static input/output lane per pending
invocation. `_reuse_graph_input_output_buffers` reuses compatible storage when
the recorded lifetimes do not overlap.

The root output hook records the graph invocations reached by each forward.
Backward validates this internal token before selecting its static lane.
`release_pending()` and runner reset advance the token epoch, so an abandoned
or pre-reset output cannot later replay against newer static buffers.
When more than one forward is pending, backward must start from an instrumented
root output so the runner can select the corresponding lane; there is no
default lane when the token is missing. A lane becomes reusable as soon as its
own backward finishes, including non-FIFO schedules where an older lane remains
pending. If all lanes are occupied, the runner asks the caller to finish
backward or call `release_pending()`.

Multi-lane inference is allowed only when the recorded training schedule is
idle. It does not advance the training replay cursor. If inference input
metadata does not match a captured static surface, the module runs its original
eager forward.

Activation-recompute graphs currently reject RNG-consuming callables. CUDA
Graph replay advances graph-safe generators but cannot restore the original
forward RNG state before `RF`, so accepting RNG would silently change dropout
or other stochastic results.

The observed schedule must remain stable across replay. Each checkpoint region
must be serial and contain a contiguous set of captured modules. Branched
regions and custom backward orders are rejected. The number of simultaneous
forwards awaiting backward is limited by
`cuda_graph_max_pending_forwards`; each additional lane has separate static
I/O and autograd state.

## Runtime pieces

| Component | Role |
| --- | --- |
| `TracePoolAllocator` | Provides stable tensor addresses for FSDP temporary buffers after the trace microbatch. |
| `CudaGraphRunner` | Records sample module inputs/outputs, invokes `make_graphed_callables()`, and installs graphed forwards. |
| `capture_time_hooks` | Run FSDP unshard/reshard during warmup and capture without recording them in the graph. |
| FSDP forward/backward hooks | Continue to run during replay around the graphed module call. |
| `te_graph_runtime` | Vendored TE-compatible graph runtime with local support needed by Megatron FSDP v2. |

## Hook behavior

During capture, real module hooks are temporarily removed because
`make_graphed_callables()` requires hook-free modules. Equivalent FSDP
unshard/reshard work is passed as `capture_time_hooks`.

The capture-time backward post-hook also releases the temporary full
`main_grad` buffer after each module graph records its address. This mirrors
the normal post-backward lifetime without launching gradient reduction during
capture. It is required because `TracePoolAllocator` may assign sequential
module-gradient keys to the same physical slot; leaving an earlier key active
would make the optimized capture lifetime differ from the eager trace.
Clone-slot detection after capture reuses the static main-gradient binding
recorded before backward. It must not call `get_main_grad()` again after the
post-hook release, because that getter can reacquire the just-released trace
slot and extend its logical lifetime into the next module capture.

During replay, real hooks are restored and fire normally:

```text
forward_pre_hook  -> unshard
graphed forward   -> replay forward graph
forward_hook      -> reshard
backward_pre_hook -> unshard for backward
graphed backward  -> replay backward graph
backward_hook     -> reshard and reduce gradients
```

## Parameter gradients

The captured backward may bind compatible parameter-gradient outputs directly
to Megatron FSDP v2 `main_grad` storage. Compatibility requires matching shape,
dtype, device, layout, stride, sharding policy, and gradient-ownership rules.

When direct binding is possible, replay writes into the optimizer-facing
gradient buffer and avoids an extra `param.grad -> main_grad` copy. When it is
not possible, the graph uses graph-owned gradient storage and FSDP copies or
accumulates into `main_grad` in the normal post-backward path.

Gradient reduction remains outside the graph in both cases.

## Requirements and limitations

- Selected modules must have static shapes, dtypes, and control flow across
  replayed microbatches.
- Selected modules must be leaf FSDP modules; nested graph-enabled FSDP modules
  are not supported.
- `TracePoolAllocator` must be used. `fully_shard(..., enable_cuda_graph=True)`
  selects it automatically.
- Releasing the trace pool invalidates captured addresses. Use
  `FSDPModule.release_memory_pool()` so graphs are dropped and recaptured before
  slot tensors are reallocated.
- Full-iteration capture uses a different runtime path; see
  [`full_iteration_cuda_graph_design.md`](full_iteration_cuda_graph_design.md).

## Relevant files

| File | Role |
| --- | --- |
| `fully_shard.py` | Selects `TracePoolAllocator` and records `enable_cuda_graph` in FSDP state. |
| `hooks.py` | Records samples, requests capture after backward, and captures at the next root forward. |
| `cuda_graph_runner.py` | Orchestrates hook save/restore and `make_graphed_callables()` invocation. |
| `te_graph_runtime/` | Vendored graph runtime used for capture and replay. |
| `trace_pool_allocator_design.md` | Details the stable-address allocator used by this path. |
