# 1F1B (EP) Overlap — FSDP Integration Design

This document describes the FSDP-side contract required by the 1F1B EP-overlap
schedule (``combined_1f1b``).  Every FSDP implementation (v1, v2, future) must
satisfy this contract for the overlap schedule to function correctly.

---

## 1. Why the overlap schedule needs special FSDP handling

**Normal FSDP flow** (hooks fire on ``TransformerLayer``):

```
TransformerLayer.forward()
  → pre-forward hook:  unshard params
  → actual compute
  → post-forward hook: reshard params
backward()
  → pre-backward hook:  unshard params
  → compute grads
  → post-backward hook: reshard params + reduce-scatter grads
```

**EP overlap flow** (calls sub-modules directly, bypassing ``TransformerLayer.forward()``):

```
combined_forward_backward_step()
  → f_layer.attn.forward()     ← no TransformerLayer hook fires
  → b_layer.mlp.backward()     ← no TransformerLayer hook fires
  → f_layer.moe_dispatch.forward()
  → ...
```

Because the schedule invokes sub-modules (``attn``, ``moe_dispatch``, ``mlp``,
``moe_combine``) directly, the FSDP hooks registered on ``TransformerLayer``
are **never triggered**.  FSDP must therefore expose a set of manual
management APIs that the schedule calls explicitly at the right moments.

---

## 2. FSDP API contract

### 2.1 Required attributes on the FSDP wrapper

| API | Type | Required for | Semantics |
|---|---|---|---|
| ``pre_backward()`` | callable | All sharding strategies | Root-level backward-phase setup. Called once before the overlapped forward+backward run. |
| ``post_backward()`` | callable | All sharding strategies | Root-level finalization. Called once after the overlapped forward+backward run. |
| ``post_forward_release_module(module)`` | callable | ``optim_grads_params`` only | Release all-gathered parameters for one layer after its forward ops complete. |
| ``post_backward_release_module(module)`` | callable | ``optim_grads_params`` only | Release all-gathered parameters for one layer after its backward ops complete. |
| ``no_sync()`` | context manager | All sharding strategies | Context manager to suppress gradient synchronization for inner micro-batches. |
| ``_replace_param_with_raw_if_needed()`` | callable | All sharding strategies | Swap optimizer-managed distributed params → raw params so the schedule accesses layers directly. |
| ``ddp_config`` | object | All | Configuration object with ``data_parallel_sharding_strategy`` attribute. |

### 2.2 Required hooks for fine-grained sub-module management

When ``overlap_moe_expert_parallel_comm=True``, two additional hook modes must
be enabled:

| Hook mode | Required for | Effect |
|---|---|---|
| Fine-grained pre-forward unshard | All strategies | Register ``_pre_forward_param_unshard`` on **every sub-module** (not just FSDP units), because the schedule calls sub-modules directly. |
| Fine-grained pre-backward unshard | ``optim_grads_params`` only | Register ``_pre_backward_param_unshard`` via ``register_multi_grad_hook`` on each FSDP unit's output tensor, ensuring params are unsharded before backward compute on that sub-module. |

### 2.3 Per-layer reshard hooks (``optim_grads_params`` only)

The schedule plan calls ``set_fsdp_reshard_hooks(post_forward_hook, post_backward_hook)``
on each ``TransformerLayerSchedulePlan`` to wire:

- **Post-forward release**: attached to the **last forward node**
  (``moe_combine`` for MoE, ``mlp`` otherwise).  Calls ``post_forward_release_module(layer)``.
- **Post-backward release**: attached to the **last backward node** (``attn``).
  Calls ``post_backward_release_module(layer)``.

These are needed because the overlap schedule bypasses ``TransformerLayer.forward()``,
so the normal ``_post_forward`` and ``_post_backward_release_module`` hooks
never fire.

---

## 3. Runtime call sequence

The overlay schedule in ``combined_forward_backward_step()`` calls FSDP APIs
in this order (file: ``megatron/core/pipeline_parallel/combined_1f1b.py``):

```
combined_1f1b_schedule_for_no_pipelining():
│
├─ 1. fsdp_wrapper._replace_param_with_raw_if_needed()    # swap params
│
├─ 2. combined_forward_backward_step(  # first microbatch, fwd only
│       f_model=model, b_model=None, ...
│     )
│     (no FSDP calls — normal TransformerLayer hooks handle this)
│
├─ 3. with no_sync_func():   # disables gradient sync for inner batches
│     │
│     └─ combined_forward_backward_step(
│          f_model=model, b_model=model, fsdp_wrapper=fsdp_wrapper, ...
│        )
│        │
│        ├─ 3a. fsdp_wrapper.pre_backward()                  # root backward setup
│        │
│        ├─ 3b. for each layer: layer_plan.set_fsdp_reshard_hooks(...
│        │       # Wires post_forward_release_module + post_backward_release_module
│        │
│        ├─ 3c. TransformerModelChunkSchedulePlan.run(...)   # overlapped fwd+bwd
│        │      # Per-layer hooks fire during the schedule plan execution:
│        │      #   - After last fwd node: post_forward_release_module(layer)
│        │      #   - After last bwd node: post_backward_release_module(layer)
│        │
│        └─ 3d. fsdp_wrapper.post_backward()                 # root finalization
│
└─ 4. combined_forward_backward_step(  # last batch, bwd only
       f_model=None, b_model=model, fsdp_wrapper=fsdp_wrapper, ...
     )
     │
     ├─ 4a. fsdp_wrapper.pre_backward()
     ├─ 4b. TransformerModelChunkSchedulePlan.run(...)
     └─ 4c. fsdp_wrapper.post_backward()
```

---

## 4. API semantics — detailed behavior

### 4.1 ``pre_backward()``

- **V1 impl**: ``_root_pre_backward(module=None, skip_backward_hook=True)``
- **When**: called once in ``combined_forward_backward_step()`` before the
  schedule plan ``.run()`` (line 339 in ``combined_1f1b.py``).
- **What it does**:
  1. Sets ``_root_pre_backward_hook_issued = True`` (idempotency guard).
  2. For ``optim_grads_params``: sets all sub-module ``_training_state`` to
     ``PRE_BACKWARD``, marks all AG buckets as releasable.
  3. Tracks params that require gradient handling via
     ``_params_require_handle_grad``.
  4. **Key**: does NOT auto-enqueue ``_root_post_backward`` — the schedule
     calls it manually via ``post_backward()``.  This is the purpose of
     ``skip_backward_hook=True``.

### 4.2 ``post_backward()``

- **V1 impl**: ``_root_post_backward()``
- **When**: called once in ``combined_forward_backward_step()`` after the
  schedule plan ``.run()`` (line 501 in ``combined_1f1b.py``).
- **What it does**:
  1. Processes any remaining unhandled gradients.
  2. Launches async reduce-scatter for gradient-sharding strategies.
  3. Resets root state: ``_root_pre_backward_hook_issued = False``,
     increments ``microbatch_count``.

### 4.3 ``post_forward_release_module(module)``

- **V1 impl**: ``_post_forward(module, input=None, output=None)``
- **When**: wired via ``set_fsdp_reshard_hooks()``, fires after the last
  forward node of each layer in the schedule plan.
- **What it does**:
  1. If ``_training_state == PRE_BACKWARD``: lazy release (activation
     recomputation case).
  2. Otherwise: release params via ``release_module_parameters(module, bwd=False)``,
     transition to ``IDLE``.

### 4.4 ``post_backward_release_module(module)``

- **V1 impl**: ``_post_backward_release_module(module)``
- **When**: wired via ``set_fsdp_reshard_hooks()``, fires after the last
  backward node (``attn``) of each layer.
- **What it does**:
  1. Releases params for both backward and forward passes:
     ``release_module_parameters(module, bwd=True)`` and
     ``release_module_parameters(module, bwd=False)``.
  2. Transitions all sub-modules to ``IDLE`` state.
  3. Gradient processing (reduce-scatter) is handled by the per-param
     ``post_accumulate_grad_hook`` that fires independently.

### 4.5 ``no_sync()``

- **V1 impl**: ``MegatronFSDP.no_sync()`` (context manager, sets
  ``is_last_microbatch = False`` on enter, ``True`` on exit).
- **When**: wraps the inner micro-batches (all except first forward-only
  and last backward-only) in ``combined_1f1b_schedule_for_no_pipelining()``.
- **Effect**: prevents gradient reduce-scatter for non-final micro-batches,
  matching the standard 1F1B gradient accumulation pattern.

### 4.6 ``_replace_param_with_raw_if_needed()``

- **V1 impl**: ``MegatronFSDP._replace_param_with_raw_if_needed()``
- **When**: called once at the start of the schedule, before any layer access.
- **Effect**: swaps the distributed (optimizer-managed) ``DTensor`` parameters
  back to raw ``nn.Parameter`` tensors so the schedule can call sub-modules
  directly.

---

## 5. Constraints enforced at init time

When ``overlap_moe_expert_parallel_comm=True``, the FSDP adapter
(``mcore_fsdp_adapter.py``) enforces:

| Constraint | Rationale |
|---|---|
| ``fsdp_double_buffer = False`` | Double buffering is incompatible with per-sub-module parameter management in the overlap schedule. |
| Only ``cuda_graph_scope = 'full'`` (or none) | Partial CUDA graph scopes conflict with the fine-grained schedule execution. |
| ``fsdp_unit_modules == [TransformerLayer]`` (for ``optim_grads_params``) | The per-layer reshard hooks assume exactly one ``TransformerLayer`` per FSDP unit. |
| Interleaved PP + FSDP is blocked | The interleaved path does not handle ``_replace_param_with_raw_if_needed`` and root pre/post-backward for multi-chunk models (assert in ``combined_1f1b.py:199``). |
| Only ``GPTModel`` is supported | The schedule plan build in ``combined_forward_backward_step()`` explicitly checks ``isinstance(unwrapped_model, GPTModel)``. |

---

## 6. Key code locations (v1 reference)

| Component | File |
|---|---|
| FSDP API definitions | ``megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`` |
| FSDP adapter (config wiring) | ``megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`` |
| Schedule orchestration | ``megatron/core/pipeline_parallel/combined_1f1b.py`` |
| Schedule plan (layer-level) | ``megatron/core/models/common/model_chunk_schedule_plan.py`` |
| Schedule dispatch | ``megatron/core/pipeline_parallel/schedules.py`` |
| Fine-grained callables | ``megatron/core/models/gpt/fine_grained_callables.py`` |
| Unit tests | ``tests/unit_tests/a2a_overlap/test_fsdp_1f1b_overlap.py`` |
| Test utilities | ``tests/unit_tests/a2a_overlap/utils.py`` |

---

## 7. V2 implementation checklist

To add this capability to Megatron FSDP v2 (``megatron_fsdp/v2/``), implement:

- [ ] **``pre_backward()``** — root-level backward phase setup that:
  - Sets backward-phase flags on the root context.
  - Optionally skips auto-enqueuing the post-backward final callback.
  - Tracks params for gradient handling.
- [ ] **``post_backward()``** — root-level finalization that:
  - Reshards + reduces gradients for any module whose per-module post-backward was skipped.
  - Waits for pending async reduce-grad events.
  - Resets root/context state for the next micro-batch.
- [ ] **``post_forward_release_module(module)``** — per-layer forward param release (``reshard()``).
- [ ] **``post_backward_release_module(module)``** — per-layer backward param release (``reshard()`` + possibly ``reduce_grad()``).
- [ ] **``_replace_param_with_raw_if_needed()``** — or equivalent param swap if v2 uses a different param management scheme.
- [ ] **``no_sync()``** — context manager to suppress gradient reduce-scatter.
- [ ] **Fine-grained pre-forward hooks** — enable sub-module-level unshard when EP overlap is active.
- [ ] **Fine-grained pre-backward hooks** — for ``optim_grads_params``, enable sub-module-level backward unshard.
- [ ] **``ddp_config.data_parallel_sharding_strategy``** — accessible for the schedule to check strategy.
- [ ] **Init-time constraint checks** — enforce ``fsdp_double_buffer=False``, CUDA graph scope, FSDP unit module compatibility.
- [ ] **Adapter wiring** — in ``mcore_fsdp_adapter.py``, expose the above APIs and enable fine-grained hooks when ``config.overlap_moe_expert_parallel_comm=True``.
