# EP Overlap + Delayed Wgrad Compute — FSDP v2 Migration Design

> **Status: Design proposal.**  None of the APIs or mechanisms described
> here are implemented yet.  Stream-level optimization (EP comm stream) is
> deferred to a later phase; this document focuses on the hook architecture.

## 1. What v1 Does

### 1.1 v1 Hook Points

v1 attaches four explicit hooks per sub-node (attn, mlp, dispatch, combine):

| Hook | Trigger | What it does |
|------|---------|--------------|
| `pre_backward` | Before the first backward starts for this micro-batch pair | FSDP setup (gradient buffer lazy allocation) |
| `post_backward` | After all backwards for this micro-batch pair complete | FSDP cleanup, reduce-scatter drain |
| `post_forward_release_module` | After a sub-node's forward completes | Release (reshard) all-gathered params of that sub-node |
| `post_backward_release_module` | After a sub-node's backward completes | Release (reshard) all-gathered params of that sub-node, transition to IDLE |

These are wired **per sub-node** in `combined_1f1b.py` via `set_fsdp_reshard_hooks()`:

```python
# combined_1f1b.py lines 390-403
for i, (f_layer_plan, b_layer_plan) in enumerate(layer_plans):
    f_post_forward_hook = functools.partial(
        fsdp_wrapper.post_forward_release_module,
        module=fsdp_module_to_release,
    )
    b_post_backward_hook = functools.partial(
        fsdp_wrapper.post_backward_release_module,
        module=fsdp_module_to_release,
    )
    f_layer_plan.set_fsdp_reshard_hooks(f_post_forward_hook, b_post_backward_hook)
```

### 1.2 How v1 Decomposes the Transformer Layer

v1 uses `TransformerLayerSchedulePlan` which breaks a transformer layer into
sub-nodes:

```
attn (comp stream)  →  dispatch (comm stream)  →  mlp (comp stream)  →  combine (comm stream)
```

Each sub-node is an independent execution unit with its own forward/backward
calls and its own FSDP release hooks.

### 1.3 delay_wgrad_compute

When enabled, TE's `Linear.backward()` skips the weight gradient kernel.
The weight gradient is computed later via `backward_dw()`, called explicitly
by the schedule plan at an overlap point.  After `backward_dw()`, hooks
trigger `_trigger_wgrad_accumulation_and_reduce_hooks()` to accumulate
and reduce-scatter the weight gradients.

## 2. v2's Current Hook Model

v2 registers hooks at **module granularity** via `fully_shard()`:

```
forward_pre_hook  →  unshard()                     (before forward)
forward_hook      →  reshard()                     (after forward)
backward_pre_hook →  unshard(bwd_pass=True), phase setup  (before backward)
backward_hook     →  reshard(), reduce_grad()      (after backward)
```

All four hooks fire for the entire FSDP module.  Parameters stay unsharded
for the duration of the module's forward, and (if ZeRO-2/3) for the duration
of the module's backward.

**The problem**: A transformer layer with MoE has both attention parameters
and expert parameters.  If they are all in one FSDP module, all parameters
stay unsharded until the entire layer's forward completes.  We cannot release
attention parameters early (after attn forward) because the hook fires once
per module, not once per sub-node.

```
Current v2 (one FSDP module = one layer):
  unshard(all) → attn → dispatch → mlp → combine → reshard(all)
                                          ↑
                        attn params still unsharded here,
                        consuming memory unnecessarily
```

## 3. Proposed Design: Fine-Grained Hooks on Sub-Modules

### 3.1 Core Idea (aligned with v1)

**Do NOT decompose the FSDP module.**  The TransformerLayer remains a single
``FSDPModule`` — all parameters (attn + mlp + layernorms) share one FSDP
param group and one buffer.  This preserves checkpoint structure, optimizer
grouping, and param-group boundaries.

Instead, **register fine-grained forward/backward hooks on the sub-modules**
(``self_attention``, ``mlp``) that manage parameter lifecycle at sub-module
granularity within the parent FSDP unit.  This is exactly how v1's
``enable_fine_grained_param_gather_hook`` works (see commit 77c0f8c).

### 3.2 Algorithm — Reference-Count Sub-Module Completions

No partial unshard/reshard API is needed.  The FSDP module unshards once
(triggered by the FIRST sub-module that needs params) and reshards once
(triggered when the LAST sub-module completes).  Reference counting tracks
completions:

```
Forward reference count:
  _submodule_fwd_done = 0           _submodule_fwd_total = 2  (attn, mlp)

  on submodule pre_forward:  if _submodule_fwd_done == 0 →  unshard(all)
  on submodule post_forward: _submodule_fwd_done += 1
                              if count == total →  reshard(all)

Backward reference count:
  _submodule_bwd_done = 0           _submodule_bwd_total = 2  (mlp, attn)

  on submodule pre_backward:  if _submodule_bwd_done == 0 →  unshard(all)
  on submodule post_backward: _submodule_bwd_done += 1
                               if count == total →  reduce_grad + reshard
```

### 3.3 Execution Timeline

```
FORWARD (one micro-batch):
  [attn pre-fwd hook]   _submodule_fwd_done==0 → unshard(all params)
  attn.forward()
  [attn post-fwd hook]  _submodule_fwd_done=1 (1/2, not yet)
  [dispatch]  (EP all-to-all, no FSDP params)
  [mlp pre-fwd hook]    _submodule_fwd_done>0 → skip (already unsharded)
  mlp.forward()
  [mlp post-fwd hook]   _submodule_fwd_done=2 (2/2) → reshard(all)

BACKWARD:
  [mlp pre-bwd hook]    _submodule_bwd_done==0 → unshard(all params)
  mlp.backward()
  [mlp post-bwd hook]   _submodule_bwd_done=1 (1/2, not yet)
  [dispatch_bwd, combine_bwd]  (EP all-to-all, no FSDP params)
  [attn pre-bwd hook]   _submodule_bwd_done>0 → skip (already unsharded)
  attn.backward()
  [attn post-bwd hook]  _submodule_bwd_done=2 (2/2) → reduce_grad + reshard
```

Only 2 unshards + 2 reshards total for the entire layer, same as the
current v2 hook model.  The only change is that unshard happens at the
first sub-module boundary instead of the FSDP module's pre_forward_hook,
and reshard happens at the last sub-module boundary instead of the FSDP
module's post_forward_hook / post_backward_hook.

### 3.4 State Tracking on FSDPModule

```python
class FSDPModule:
    # Added when enable_ep_overlap() is called
    _ep_submodule_fwd_total: int = 0  # e.g., 2 (self_attention, mlp)
    _ep_submodule_fwd_done: int = 0
    _ep_submodule_bwd_total: int = 0
    _ep_submodule_bwd_done: int = 0

    def _ep_on_submodule_pre_forward(self, submodule):
        if self._ep_submodule_fwd_done == 0:
            self.unshard(bwd_pass=False)
        # Transition training state for the sub-module

    def _ep_on_submodule_post_forward(self, submodule):
        self._ep_submodule_fwd_done += 1
        if self._ep_submodule_fwd_done == self._ep_submodule_fwd_total:
            self._ep_submodule_fwd_done = 0
            self.reshard()

    def _ep_on_submodule_pre_backward(self, submodule):
        if self._ep_submodule_bwd_done == 0:
            self.unshard(bwd_pass=True)
            self.unshard(bwd_pass=False)

    def _ep_on_submodule_post_backward(self, submodule):
        self._ep_submodule_bwd_done += 1
        if self._ep_submodule_bwd_done == self._ep_submodule_bwd_total:
            self._ep_submodule_bwd_done = 0
            self.reshard()
            self.reduce_grad()  # all sub-module grads done
```

### 3.5 Comparison with v1

| Concern | v1 | v2 (proposed) |
|---------|----|---------------|
| FSDP unit | One MegatronFSDP wrapper per layer | One FSDPModule per layer (unchanged) |
| Fine-grained hooks | `_register_pre_forward_param_unshard_hook` on sub-modules | Pre/post forward/backward hooks on sub-modules |
| Unshard trigger | Per sub-module via fine-grained hooks | First sub-module pre-forward/backward (ref count == 0) |
| Reshard trigger | `post_forward_release_module(module)` | Last sub-module post-forward/backward (ref count == total) |
| Grad reduce trigger | `post_backward_release_module(module)` | Last sub-module post-backward |
| Schedule wiring | `set_fsdp_reshard_hooks()` explicit | Automatic via PyTorch hook dispatch on sub-modules |
| FSDP buffer | One buffer per layer (unchanged) | One buffer per layer (unchanged) |
| Checkpoint structure | Unchanged | Unchanged |

### 3.6 Why Reference Counting Is Sufficient

The entire FSDP buffer must be unsharded for ANY sub-module to access its
params, because params are interleaved in a flat buffer.  There is no
benefit to partial unshard — the same all-gather happens regardless.

Reference counting defers reshard from the FSDP-module-level hook to the
last sub-module-level hook.  Total unshard/reshard operations per
forward+backward pass are unchanged (2 unshards + 2 reshards).

## 4. delay_wgrad_compute Integration

When `delay_wgrad_compute` is enabled for MoE layers, the expert weight
gradient computation is moved from `backward()` to `backward_dw()`.
In the v2 hook model, this interacts with `reduce_grad()`:

**Problem**: `reduce_grad()` is called in `backward_hook` after `backward()`
completes.  If `backward()` skipped weight gradients, the gradient buffer
is empty at that point.  `reduce_grad()` must be deferred until after
`backward_dw()` has populated the buffer.

**Solution**: Add a `_deferred_reduce_grads` list to the sub-module's
`_FSDPState` (or `ParameterGroup`).  When `delay_wgrad_compute` is active:

1. `backward_hook` skips `reduce_grad()` and instead marks the parameter
   group as having a deferred reduce.
2. After `backward_dw()` writes weight gradients to `main_grad`, it
   triggers the deferred `reduce_grad()`.
3. Any remaining deferred reduces are drained in the final callback.

```python
# In FSDPModule or ParameterGroup:
def _defer_reduce_grad(self):
    """Called by backward_hook instead of reduce_grad()."""
    self._reduce_grad_deferred = True

def _drain_deferred_reduce_grad(self):
    """Called after backward_dw() or in final callback."""
    if self._reduce_grad_deferred:
        self.reduce_grad(async_op=False)
        self._reduce_grad_deferred = False
```

### 4.1 TE backward_dw() Trigger

In v1, `backward_dw()` is called by the schedule plan at specific overlap
points.  In v2, we can:

**Option A (simple)**: Call `module.backward_dw()` for each MoE sub-module
in the final callback, after all backwards complete.  This is correct but
doesn't overlap.

**Option B (overlap)**: Register a callback on the autograd graph that
triggers `backward_dw()` when the sub-module's backward completes.  This
is equivalent to v1's schedule but uses PyTorch's autograd hooks instead
of a custom schedule plan:

```python
def _register_backward_dw_callback(sub_module):
    """After sub_module.backward() completes, trigger backward_dw() on the
    MoE expert module, then drain deferred reduce_grad."""
    def _dw_callback(grad_outputs):
        if hasattr(sub_module, 'backward_dw'):
            sub_module.backward_dw()
        sub_module._drain_deferred_reduce_grad()
    # Register on the output tensor's grad_fn
    ...
```

## 5. Nested FSDP Handling

When fine-grained hooks are used, the FSDP module hierarchy does NOT change.
The TransformerLayer is still one FSDPModule.  Fine-grained hooks on
sub-modules are just additional callbacks within the same FSDP unit.

```
model (FSDP root)
  └── layers.0 (FSDP)               ← still one FSDP module
        ├── self_attention          ← has fine-grained hooks
        ├── [dispatch] (pure comm)
        └── mlp                     ← has fine-grained hooks
  └── layers.1 (FSDP)
        ├── self_attention
        └── mlp
```

The root context is unchanged.  Phase tracking (forward_phase, backward_phase)
and the final callback remain managed by the existing hook system.

## 6. Implementation Plan

### Phase 1 — Reference-counted sub-module hooks (no delay_wgrad)

1. Add `_ep_submodule_fwd/bwd_total` and `_ep_submodule_fwd/bwd_done`
   counters to `FSDPModule` (or `_FSDPState`).
2. When EP overlap is active, register pre/post forward/backward hooks
   on `layer.self_attention` and `layer.mlp`.
3. Pre-hook: if count==0 → `unshard()`. Post-hook: increment; if done → `reshard()`.
4. Disable the existing FSDP-module-level `forward_pre_hook` and
   `forward_hook` for this module (sub-module hooks take over).
5. Verify forward correctness (same loss as without EP overlap).

### Phase 2 — Backward sub-module hooks + reduce_grad

1. Add backward pre/post hooks on sub-modules with same reference-counting pattern.
2. Post-hook: if all backwards done → `reduce_grad()` before `reshard()`.
3. Verify backward correctness (same gradients as without EP overlap).

### Phase 3 — delay_wgrad_compute

1. Add `_deferred_reduce_grads` tracking to ParameterGroup.
2. When `delay_wgrad_compute` is active, `reduce_grad()` is deferred
   until after `backward_dw()` populates the gradient buffer.
3. Wire TE `backward_dw()` to trigger deferred reduce.

### Phase 4 — 1F1B pipeline + EP comm stream

1. Add `ep_comm_stream` to `_FSDPRootContext`.
2. Route MoE dispatch/combine kernels to the EP stream.
3. Verify correctness with PP>1 1F1B scheduling.
4. Benchmark against v1.
