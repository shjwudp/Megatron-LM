# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FSDPModule implementation for Megatron-FSDP2."""

import logging
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from .allocator import BucketAllocator, TracePoolAllocator
from .mixed_precision import MixedPrecisionPolicy
from .param_group import ParameterGroup, ParameterGroupLayout
from .utils import ParamGroupIdx, _replace_module_parameter

logger = logging.getLogger(__name__)


class _FSDPState:
    """
    Internal state for FSDP module tracking.

    Attributes:
        _is_root: Whether this is the root FSDP module (handles final callback).
        _post_backward_callback_queued: Whether callback is queued for execution.
    """

    def __init__(self):
        self._is_root = True
        self._post_backward_callback_queued = False
        self.enable_cuda_graph: bool = False
        self.enable_full_iteration_cuda_graph: bool = False


@dataclass
class _FSDPRootContext:
    """
    Runtime context shared across all FSDP modules within a single root.

    This object coordinates CUDA streams, execution ordering, and async
    communication overlap (all-gather / reduce-scatter) during forward
    and backward passes.
    """

    # ------------------------------------------------------------------
    # CUDA streams (communication overlap)
    # ------------------------------------------------------------------
    ag_streams: Tuple[torch.cuda.Stream, ...]  # one all-gather stream per mesh axis
    rs_streams: Tuple[torch.cuda.Stream, ...]  # one reduce-scatter stream per mesh axis

    @property
    def rs_stream(self) -> torch.cuda.Stream:
        """Return the first HSDP reduction stream for shared-stream call sites."""
        return self.rs_streams[-1]

    # ------------------------------------------------------------------
    # Bucket allocator
    # ------------------------------------------------------------------
    bucket_allocator: BucketAllocator
    """
    Bucket allocator for temporary all-gather and reduce-scatter buffers.

    ParameterGroup buffer roles are part of each allocation key, so one
    allocator can safely manage all temporary leases without separate
    weight/gradient allocator instances.
    """

    # ------------------------------------------------------------------
    # Forward execution ordering
    # ------------------------------------------------------------------
    forward_order: List["FSDPModule"] = field(default_factory=list)
    """
    FSDP modules in actual forward execution order.

    This ordering is used to:
    - Schedule prefetching of parameters (unshard)
    - Ensure correct overlap between compute and communication
    """

    # ------------------------------------------------------------------
    # Unshard (all-gather) tracking
    # ------------------------------------------------------------------
    unshard_done_events: Dict[int, Optional[torch.cuda.Event]] = field(default_factory=dict)
    """
    Maps module_id -> CUDA event signaling completion of parameter unshard.

    Used to enforce correct dependency between all-gather and compute.
    """

    enable_unshard_prefetch: bool = True
    """Whether to prefetch (pipeline) parameter unshard for upcoming modules."""

    # ------------------------------------------------------------------
    # Reduce-scatter (gradient sync) tracking
    # ------------------------------------------------------------------
    reduce_grad_buckets: Dict[int, List[Tuple[torch.cuda.Event, "ParameterGroup"]]] = field(
        default_factory=dict
    )
    """
    Maps module_id -> list of (event, parameter_group) tuples.

    Each entry corresponds to a module and contains a list of:
        (event, parameter_group)

    - event: signals gradient readiness
    - parameter_group: gradients to be reduced

    This structure enables ordered overlap of backward compute and
    gradient synchronization.
    """

    enable_async_reduce_grad: bool = True
    """Whether to overlap gradient reduction with backward computation."""

    is_last_backward: bool = False
    """Whether the current backward pass is the optimizer-step boundary."""

    model_weight_refresh_pending: bool = False
    """Whether the next normal forward must install optimizer-updated weights.

    The final backward callback arms this only at an optimizer-step boundary.
    An explicit post-optimizer copy or the next non-recompute forward consumes it.
    """

    # ------------------------------------------------------------------
    # Activation recompute / gradient checkpointing support
    # ------------------------------------------------------------------
    backward_phase: bool = False
    """True from the root backward pre-hook until the final callback.
    ``forward_phase`` is set to ``False`` when this becomes ``True``."""

    forward_phase: bool = False
    """True from the root forward pre-hook until the root backward pre-hook.
    ``backward_phase`` is set to ``False`` when this becomes ``True``."""

    enable_cuda_graph: bool = False
    """Whether hooks should manage the side stream for CUDA graph capture."""

    cuda_graph_stream: Optional[torch.cuda.Stream] = None
    """Side stream for CUDA graph capture/replay.  Created lazily on the
    first forward pre-hook and shared across all FSDP modules."""

    cuda_graph_active: bool = False
    """True while ``make_graphed_callables`` is inside its capture
    region.  All hooks must assert ``not cuda_graph_active`` — hooks
    are popped during capture, so a callback firing inside the capture
    window indicates a bug."""

    cuda_graph_pool: Optional[Any] = None
    """Shared CUDA graph memory pool handle for CUDA graph capture."""

    cuda_graph_runner: Optional[Any] = None
    """``CudaGraphRunner`` instance.  Created lazily on the first
    optimized forward pre-hook and reused across micro-batches."""

    backward_module: Optional[int] = None
    """``id(module)`` of the FSDP module whose backward is pending next.
    Derived from ``_reversed_order`` and ``backward_done_modules`` — NOT
    set by any hook directly.  Updated by ``_advance_backward_module()``."""

    backward_done_modules: set = field(default_factory=set)
    """Set of ``id(module)`` for FSDP modules whose backward has completed.
    Populated in ``post_backward``, cleared in the root backward pre-hook."""

    _reversed_order: List["FSDPModule"] = field(default_factory=list)
    """``list(reversed(forward_order))`` — precomputed backward processing order."""

    def _advance_backward_module(self) -> None:
        """Set ``backward_module`` to the first module in ``_reversed_order``
        that is NOT in ``backward_done_modules``."""
        for m in self._reversed_order:
            if id(m) not in self.backward_done_modules:
                self.backward_module = id(m)
                return
        self.backward_module = None

    def get_prefetch_next_modules(
        self, module: "FSDPModule", bwd_pass: bool = False
    ) -> List["FSDPModule"]:
        """Return the next FSDP module to prefetch in forward or backward order."""
        module_order = list(reversed(self.forward_order)) if bwd_pass else self.forward_order

        for module_index, candidate_module in enumerate(module_order):
            if candidate_module is module:
                if module_index + 1 >= len(module_order):
                    return []
                return [module_order[module_index + 1]]

        raise AssertionError("Current module not found in forward module order")

    def get_root_module(self):
        """Return the root FSDP module associated with this context."""
        return self.forward_order[0] if self.forward_order else None


class FSDPModule:
    """
    Mixin class for FSDP-wrapped modules.

    This class is dynamically added to wrapped modules and provides
    methods for managing parameter sharding state:
    - unshard(): All-gather parameters before forward
    - reshard(): Release unsharded buffer after forward
    - reduce_grad(): Reduce gradients after backward
    """

    @property
    def cuda_graph_compatible(self) -> bool:
        """Return True when the root context is configured for CUDA graph capture.

        Requires side-stream collectives to be disabled so every CUDA
        operation lands on the default stream.  Can be used as a guard
        before entering a graph capture region::

            assert module.cuda_graph_compatible
        """
        ctx = self._fsdp_root_context
        if not isinstance(ctx.bucket_allocator, TracePoolAllocator):
            return False
        if ctx.bucket_allocator.phase != "optimized":
            return False
        return True

    def release_memory_pool(self) -> None:
        """Release all persistent communication-buffer memory and any CUDA graphs.

        Tears down captured CUDA graphs across all FSDP modules, clears graph
        sentinels so they auto-recapture on the next forward pass, and releases
        the ``TracePoolAllocator`` slot tensors.

        On the next ``allocate`` / ``free`` call the allocator **automatically**
        re-allocates slots, so no explicit "resume" call is needed.  CUDA graphs
        are re-captured by the hooks on the next forward pass.

        Typical use: temporarily free GPU memory (e.g. for checkpoint I/O).
        """
        ctx = self._fsdp_root_context
        allocator = ctx.bucket_allocator
        for module in self._get_fsdp_modules(recursive=True):
            for pg in module._fsdp_param_groups:
                pg.release_grad_buffer()

        if not isinstance(allocator, TracePoolAllocator):
            return

        self._release_cuda_graphs(ctx)
        self._clear_cuda_graph_sentinels(ctx)
        allocator.release()

    # ----------------------------------------------------------------
    # Internal: CUDA graph teardown / sentinel helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _release_cuda_graphs(ctx: "_FSDPRootContext") -> None:
        """Tear down all captured CUDA graphs on every FSDP module.

        Supports both the per-module runner path (``_fsdp_cg_runner``) and
        the batch helper path (``_fsdp_cuda_graphs``, ``_fsdp_cg_runner`` sentinel).
        Restores original ``forward`` methods before deleting graph objects.
        """
        if not ctx.enable_cuda_graph:
            return

        for module in ctx.forward_order:
            if hasattr(module, "_fsdp_cg_runner"):
                runner = module._fsdp_cg_runner
                if hasattr(runner, "reset"):
                    runner.reset()
                delattr(module, "_fsdp_cg_runner")

        ctx.cuda_graph_active = False
        ctx.cuda_graph_stream = None
        ctx.cuda_graph_pool = None

    @staticmethod
    def _clear_cuda_graph_sentinels(ctx: "_FSDPRootContext") -> None:
        """Clear CUDA graph sentinels so hooks will re-capture on next forward."""
        for module in ctx.forward_order:
            if hasattr(module, "_fsdp_cg_runner"):
                delattr(module, "_fsdp_cg_runner")

    # ----------------------------------------------------------------
    # CPU offload
    # ----------------------------------------------------------------

    def _get_fsdp_modules(self, recursive: bool = True) -> List["FSDPModule"]:
        """Return ``[self]`` plus optionally all child ``FSDPModules``."""
        if not recursive:
            return [self]
        result = [self]
        for _, child in self.named_modules():
            if child is not self and isinstance(child, FSDPModule):
                result.append(child)
        return result

    def offload_to_cpu(
        self, recursive: bool = True, pin_memory: bool = False, max_cpu_bytes: Optional[int] = None
    ) -> Dict[str, int]:
        """Raise because CPU offload is not supported."""
        _ = recursive, pin_memory, max_cpu_bytes
        raise NotImplementedError("ParameterGroup CPU offload is not implemented yet")

    def reload_to_gpu(self, recursive: bool = True) -> None:
        """Raise because CPU reload is not supported."""
        _ = recursive
        raise NotImplementedError("ParameterGroup CPU reload is not implemented yet")

    def _init_named_param_groups(
        self,
        mesh: Optional[DeviceMesh],
        ignored_params: Optional[set],
        mp_policy: MixedPrecisionPolicy,
        gradient_scaling_factor: Optional[float] = None,
        sharding_strategy: str = "optim_grads_params",
        outer_dp_sharding_strategy: str = "no_shard",
    ):
        """
        Initialize parameter groups and build param name mapping.

        This method:
        1. Collects ignored modules (nested FSDP modules)
        2. Materializes meta modules to actual devices
        3. Groups parameters by (device, dtype, requires_grad)
        4. Builds parameter name to parameter mapping
        """
        ignored_params = ignored_params or set()
        ignored_modules = set()

        # Collect nested FSDP modules as ignored
        for _, child in self.named_modules():
            if child is not self and isinstance(child, FSDPModule):
                ignored_params.update(child.parameters())
                for child_submodule in child.modules():
                    ignored_modules.add(child_submodule)

        # Materialize meta parameters to actual device
        self._materialize_meta_module(ignored_modules, mesh=mesh, mp_policy=mp_policy)

        # Create parameter groups
        fsdp_param_groups = _get_module_fsdp_param_groups(
            self,
            mp_policy=mp_policy,
            mesh=mesh,
            ignored_params=ignored_params,
            gradient_scaling_factor=gradient_scaling_factor,
            sharding_strategy=sharding_strategy,
            outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        )
        setattr(self, "_fsdp_param_groups", fsdp_param_groups)

        # Build param name to param mapping for later lookup
        param_to_name = {p: n for n, p in self.named_parameters()}
        self._named_param_groups = []

        for fsdp_param_group in fsdp_param_groups:
            param_names = []
            for param in fsdp_param_group.params:
                param_name = param_to_name[param]
                param_names.append(param_name)
            self._named_param_groups.append((param_names, fsdp_param_group))

    def _init_param_main_grad_func(self):
        """
        Initialize main gradient getter function for each parameter.

        This creates a closure that fetches the gradient from the
        gradient buffer when accessed. It handles both sharded and
        unsharded gradient buffers.
        """

        def main_grad_getter(p):
            """Get main gradient from buffer with proper offset/size."""
            return p._fsdp_param_group.get_main_grad(p)

        # Attach getter to each parameter
        for param_group in self._fsdp_param_groups:
            for param in param_group.params:
                setattr(param, "_fsdp_param_group", param_group)
                setattr(param, "_gbuf", param_group.grad_buffer)
                setattr(param, "_item_id", param_group.param_idx[param])
                param.get_main_grad = main_grad_getter.__get__(param)

    def _materialize_meta_module(
        self,
        ignored_modules: set,
        mesh: Optional[DeviceMesh] = None,
        mp_policy: Optional[MixedPrecisionPolicy] = None,
    ):
        """
        Materialize meta parameters to actual device and initialize.

        This is needed for large models that cannot fit in a single GPU.
        Meta parameters are moved to the current device and reset.
        After materialization, full parameters are broadcast from DP rank 0
        before DTensor wrapping so every rank shards the same initialized value.
        """
        current_device = torch.cuda.current_device()
        # The CUDA API returns an integer in production. Accept a torch.device override
        # so CPU-only materialization tests can exercise this path without a GPU.
        materialization_device = (
            current_device
            if isinstance(current_device, torch.device)
            else torch.device("cuda", current_device)
        )
        # Initialize leaves before parents so a parent's reset hook may safely
        # inspect or derive state from already-materialized descendants.
        for name, m in reversed(list(self.named_modules())):
            if m in ignored_modules:
                continue
            # Match v1 meta init: reset modules that own meta parameters. Buffer-only
            # meta modules may intentionally keep lazy state initialized in forward.
            if any(p.is_meta for p in m.parameters(recurse=False)):
                m._apply(
                    lambda t: (
                        torch.empty_like(t, device=materialization_device) if t.is_meta else t
                    ),
                    recurse=False,
                )
                init_context = (
                    mp_policy.model_init_context(m) if mp_policy is not None else nullcontext()
                )
                with init_context:
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()
                    elif hasattr(m, "_reset_parameters"):
                        m._reset_parameters()
                    else:
                        raise ValueError(
                            f"Module {name} contains meta parameters but cannot reset them"
                        )

            # Move only this module's direct tensors. named_modules() visits each child
            # separately, so it is moved only after its own materialization and reset.
            # Buffer-only modules may intentionally keep lazy meta state for forward.
            # Move materialized tensors, but do not initialize those buffers here.
            m._apply(lambda t: t if t.is_meta else t.to(materialization_device), recurse=False)

        if mesh is not None and mesh.size() > 1:
            for param in self.parameters():
                if param.is_meta or isinstance(param, DTensor):
                    continue
                for mesh_dim in range(mesh.ndim):
                    group = mesh.get_group(mesh_dim=mesh_dim)
                    if torch.distributed.get_world_size(group) == 1:
                        continue
                    src_rank = torch.distributed.get_global_rank(group, 0)
                    torch.distributed.broadcast(param.data, src=src_rank, group=group)

    def _init_fsdp_state(
        self,
        enable_unshard_prefetch,
        enable_async_reduce_grad,
        mesh_ndim: int,
        all_gather_streams: Sequence[torch.cuda.Stream | None] | None,
        reduce_scatter_streams: Sequence[torch.cuda.Stream | None] | None,
        bucket_allocator: BucketAllocator,
        enable_cuda_graph: bool = False,
        enable_full_iteration_cuda_graph: bool = False,
    ):
        """Initialize FSDP state and mark nested FSDP modules as non-root.

        Important: This must be called BEFORE any forward/backward pass runs.
        Re-initializing while child FSDPModules are actively unsharded
        (mid-forward or mid-backward) will corrupt their state.  The safety
        check below enforces that constraint.
        """
        named_forward_modules = [
            (name, child) for name, child in self.named_modules() if isinstance(child, FSDPModule)
        ]
        forward_order = [child for name, child in named_forward_modules]

        # Safety check: no child FSDPModule must be in an active state.
        # - unshard_done_events[id(child)] non-None → unsharded, not yet resharded
        # - reduce_grad_buckets[id(child)] non-empty → reduce-scatter in flight
        # Re-initializing _fsdp_root_context while a child is in either state
        # would overwrite its shared state mid-pass.
        for child_module in forward_order:
            if child_module is self:
                continue
            if not hasattr(child_module, "_fsdp_root_context"):
                continue
            ctx = child_module._fsdp_root_context
            if ctx is None:
                continue
            if ctx.unshard_done_events.get(id(child_module)) is not None:
                raise RuntimeError(
                    "_init_fsdp_state cannot be called while a child FSDPModule "
                    "is still unsharded. All children must be resharded before "
                    "re-initializing FSDP state."
                )
            if ctx.reduce_grad_buckets.get(id(child_module)):
                raise RuntimeError(
                    "_init_fsdp_state cannot be called while a child FSDPModule "
                    "has pending reduce-scatter operations. All children must have "
                    "completed gradient reduction before re-initializing FSDP state."
                )

        def resolve_axis_streams(
            configured: Sequence[torch.cuda.Stream | None] | None, overlap: bool
        ) -> Tuple[torch.cuda.Stream, ...]:
            caller_stream = torch.cuda.current_stream()
            if configured is not None:
                if len(configured) != mesh_ndim:
                    raise ValueError(f"Expected {mesh_ndim} streams, got {len(configured)}")
                return tuple(stream or caller_stream for stream in configured)
            shared_stream = torch.cuda.Stream() if overlap else caller_stream
            return (shared_stream,) * mesh_ndim

        ag_streams = resolve_axis_streams(all_gather_streams, enable_unshard_prefetch)
        rs_streams = resolve_axis_streams(reduce_scatter_streams, enable_async_reduce_grad)
        root_context = _FSDPRootContext(
            ag_streams=ag_streams,
            rs_streams=rs_streams,
            forward_order=forward_order,
            reduce_grad_buckets={id(module): [] for module in forward_order},
            unshard_done_events={id(module): None for module in forward_order},
            enable_unshard_prefetch=enable_unshard_prefetch,
            enable_async_reduce_grad=enable_async_reduce_grad,
            _reversed_order=list(reversed(forward_order)),
            bucket_allocator=bucket_allocator,
        )
        setattr(self, "_fsdp_state", _FSDPState())
        self._fsdp_state.enable_full_iteration_cuda_graph = enable_full_iteration_cuda_graph
        setattr(self, "_fsdp_root_context", root_context)

        module_idx = 0
        for name, module in named_forward_modules:
            module._fsdp_state.enable_full_iteration_cuda_graph = enable_full_iteration_cuda_graph
            for param_group in module._fsdp_param_groups:
                param_group.set_allocator(root_context.bucket_allocator)
                param_group.enable_full_iteration_cuda_graph = enable_full_iteration_cuda_graph

            if module is not self:
                module._fsdp_state._is_root = False
                setattr(module, "_fsdp_root_context", root_context)

            setattr(module, "_fsdp_module_idx", module_idx)
            setattr(module, "_fsdp_module_name", name)
            module._fsdp_pre_backward_done = False
            module.post_backward_issued = False
            module_idx += 1

        # Annotate every non-FSDPModule sub-module with its nearest parent
        # FSDPModule.  Process bottom-up (reverse forward_order) so that
        # child FSDPModules claim their sub-modules before the root reaches
        # them.
        for module in reversed(forward_order):
            for submodule in module.modules():
                if isinstance(submodule, FSDPModule):
                    continue
                if hasattr(submodule, '_fsdp_parent_module'):
                    continue
                submodule._fsdp_parent_module = weakref.ref(module)

        if enable_cuda_graph:
            if len(forward_order) > 1:
                child_names = [name for name, m in named_forward_modules if m is not self]
                raise RuntimeError(
                    f"enable_cuda_graph=True is not supported for FSDP modules that contain "
                    f"other FSDP modules as children. "
                    f"Module '{self._fsdp_module_name}' (type={type(self).__name__}) "
                    f"has FSDP children: {child_names}. "
                    f"Only leaf FSDP modules (no FSDP children) can use CUDA graph capture."
                )
            self._fsdp_state.enable_cuda_graph = True

        if any(module._fsdp_state.enable_cuda_graph for module in forward_order):
            root_context.enable_cuda_graph = True

    def unshard(self, async_op: bool = False, bwd_pass: bool = False, prefetch: bool = True):
        """
        Unshard parameters by all-gathering from the sharded buffer.

        This is called pre-forward to make parameters available for
        computation. After unsharding, each param.data points to
        the full (unsharded) tensor.
        """
        torch.cuda.nvtx.range_push("MFSDP unshard")
        ctx = self._fsdp_root_context
        caller_stream = torch.cuda.current_stream()
        # Unshard this module and optionally prefetch next modules in the forward/backward pass
        if async_op and prefetch:
            prefetch_modules = ctx.get_prefetch_next_modules(self, bwd_pass=bwd_pass)
        else:
            prefetch_modules = []
        for module in [self] + prefetch_modules:
            if all(
                param_group.weights_are_unsharded(bwd_pass=bwd_pass)
                for param_group in module._fsdp_param_groups
            ):
                continue
            if bwd_pass and id(module) in ctx.backward_done_modules:
                continue  # Skip prefetch for modules whose backward is already done

            for param_names, param_group in module._named_param_groups:
                # Optional NaN checking for debugging
                if getattr(module, "_enable_nan_checks", False):
                    for name, dist_param in zip(param_names, param_group.optimizer_params):
                        assert not torch.isnan(
                            dist_param._local_tensor
                        ).any(), f"NaN detected in dist param for parameter {name}"

            streams = (
                ctx.ag_streams
                if async_op
                else (caller_stream,) * module._fsdp_param_groups[0].mesh.ndim
            )
            ParameterGroup.unshard_weights(
                module._fsdp_param_groups, streams=streams, async_op=async_op
            )

            # Record event to track when unshard is done for this module
            if async_op:
                event = ctx.ag_streams[-1].record_event()
                ctx.unshard_done_events[id(module)] = event

        # Ensure unshard is complete before forward.
        # The event is NOT cleared here — it persists as a "currently unsharded"
        # flag and is only cleared by reshard().  This prevents redundant
        # all-gathers during activation recompute and prefetch re-entry.
        if ctx.unshard_done_events[id(self)] is not None:
            ctx.unshard_done_events[id(self)].wait()

        # Replace module parameters with unsharded versions
        for param_names, param_group in self._named_param_groups:
            for name, param in zip(param_names, param_group.params):
                _replace_module_parameter(self, name, param)

            # Optional NaN checking for debugging
            if getattr(self, "_enable_nan_checks", False):
                for name, param in zip(param_names, param_group.params):
                    assert not torch.isnan(param).any(), f"NaN detected in parameter {name}"

        torch.cuda.nvtx.range_pop()

    def reshard(self):
        """Reshard parameters by replacing with sharded DTensors."""
        torch.cuda.nvtx.range_push("MFSDP reshard")
        ctx = self._fsdp_root_context
        unshard_event = ctx.unshard_done_events[id(self)]
        if unshard_event is not None:
            # A prefetched module may be skipped by control flow. Join its
            # communication before releasing caller-owned temporary buffers.
            unshard_event.wait()
        for param_names, param_group in self._named_param_groups:
            param_group.reshard_weight()
            for name, dist_param in zip(param_names, param_group.optimizer_params):
                _replace_module_parameter(self, name, dist_param)
        ctx.unshard_done_events[id(self)] = None  # Clear unshard event for this module
        torch.cuda.nvtx.range_pop()

    def _wait_for_previous_async_reduce_grad(self):
        """Release older async reduce buffers in backward order."""
        ctx = self._fsdp_root_context
        if not ctx.enable_async_reduce_grad:
            return

        backward_order = list(reversed(ctx.forward_order))
        for i, module in enumerate(backward_order):
            if i - 2 >= 0:
                buckets = ctx.reduce_grad_buckets[id(backward_order[i - 2])]
                while len(buckets) > 0:
                    event, param_group = buckets.pop()
                    event.wait()
                    param_group.release_grad_buffer()
            if module is self:
                break

    def reduce_grad(self, async_op: bool = False):
        """
        Reduce gradients across data-parallel ranks.

        This is called post-backward to:
        1. Copy gradients to main gradient buffer
        2. Perform gradient reduction
        3. Install reduced gradients to distributed parameters
        """
        torch.cuda.nvtx.range_push("MFSDP reduce_grad")
        ctx = self._fsdp_root_context
        caller_stream = torch.cuda.current_stream()
        stream = ctx.rs_stream if async_op else caller_stream

        # Handle pending reduce events before this module to release buffers promptly.
        self._wait_for_previous_async_reduce_grad()

        # Perform reduction for this module
        for param_names, param_group in self._named_param_groups:
            if not param_group.requires_grad:
                continue

            # Materialize optimizer-gradient storage and DTensor views if needed.
            param_group.prepare_gradient_storage()

            # NaN check before reduction
            if getattr(self, "_enable_nan_checks", False):
                for name, param in zip(param_names, param_group.params):
                    if param.grad is not None:
                        assert not torch.isnan(
                            param.grad
                        ).any(), f"NaN in parameter grad for {name}"

            # Stage .grad into the main grad buffer before reduce-scatter.
            # When gradient_accumulation_fusion is active for FSDP params, the backward
            # kernel writes directly into main_grad (weight.main_grad = get_main_grad() in
            # layers.py) and sets grad_added_to_main_grad=True. In that case we must NOT
            # zero or overwrite main_grad; discard the dummy .grad tensor if present.
            #
            # Under CUDA graph replay, TE's pure-GPU backward kernel still runs, but
            # the Python-side ``setattr(param, "grad_added_to_main_grad", True)`` that
            # accompanies the eager backward is captured away.  We record the per-param
            # flag during the trace micro-batch and restore it here.
            accumulate_full_grad = param_group.full_grad_has_value
            stage_tensors: List[torch.Tensor] = []
            stage_sources: List[torch.Tensor] = []
            zero_tensors: List[torch.Tensor] = []
            params_with_grad = []

            for param in param_group.params:
                grad = param.grad
                if grad is not None:
                    params_with_grad.append(param)
                grad_added = getattr(param, "grad_added_to_main_grad", False)
                recorded = getattr(param, "_mfsdp_recorded_te_wgrad", False)

                if grad_added or recorded:
                    if param.grad is not None:
                        del param.grad
                    # Record TE wgrad-fusion flags for CUDA graph restore.
                    # The trace backward ran eagerly, so TE set
                    # grad_added_to_main_grad on each param it wrote to.
                    # Under CUDA graph replay only the GPU kernel runs;
                    # we record the flags here and restore them in
                    # the CG replay backward.
                    if grad_added and self._fsdp_state.enable_cuda_graph:
                        setattr(param, "_mfsdp_recorded_te_wgrad", True)
                elif grad is None:
                    if not accumulate_full_grad:
                        zero_tensors.append(param.get_main_grad())
                else:
                    main_grad = param.get_main_grad()
                    if grad.data_ptr() != main_grad.data_ptr():
                        stage_tensors.append(main_grad)
                        stage_sources.append(grad.detach())

            # Full-iteration graphs stage ordinary async gradients on the RS stream so
            # the add/copy/zero work overlaps with the next module's backward compute.
            stage_on_rs_stream = async_op and getattr(
                self._fsdp_state, "enable_full_iteration_cuda_graph", False
            )
            if stage_on_rs_stream:
                stream.wait_stream(torch.cuda.current_stream())
                for source in stage_sources:
                    if source.is_cuda:
                        source.record_stream(stream)
                with torch.cuda.stream(stream):
                    if stage_tensors:
                        if accumulate_full_grad:
                            torch._foreach_add_(stage_tensors, stage_sources)
                        else:
                            torch._foreach_copy_(stage_tensors, stage_sources)
                    if zero_tensors:
                        torch._foreach_zero_(zero_tensors)
            else:
                if stage_tensors:
                    if accumulate_full_grad:
                        torch._foreach_add_(stage_tensors, stage_sources)
                    else:
                        torch._foreach_copy_(stage_tensors, stage_sources)
                if zero_tensors:
                    torch._foreach_zero_(zero_tensors)

            for param in params_with_grad:
                if param.grad is not None:
                    del param.grad

            stage_tensors.clear()
            stage_sources.clear()
            zero_tensors.clear()
            grad = None

            for param in param_group.params:
                # Consume this per-backward marker here. A skipped module may not run
                # _pre_backward_setup on the next microbatch, so leaving it set would
                # make stale scratch storage look like a fused wgrad.
                param.grad_added_to_main_grad = False

            if async_op:
                # ---- Overlapped path ----
                # Switch to rs_stream for the reduce-scatter kernel
                completion_stream = param_group.reduce_grad(
                    is_last_backward=ctx.is_last_backward, streams=ctx.rs_streams, async_op=True
                )
            else:
                # ---- Non-overlapped path ----
                # Reduce gradients immediately and release grad buffer
                completion_stream = param_group.reduce_grad(is_last_backward=ctx.is_last_backward)
                param_group.release_grad_buffer()

            # Install reduced gradients to distributed parameters
            for name, param, dist_param, dist_grad in zip(
                param_names,
                param_group.params,
                param_group.optimizer_params,
                param_group.optimizer_grads,
            ):
                if not param.requires_grad:
                    continue
                if param_group.mp_policy.use_decoupled_grad:
                    setattr(dist_param, "decoupled_grad", dist_grad)
                    if param_group.enable_full_iteration_cuda_graph and dist_grad is not None:
                        setattr(dist_param, "_mfsdp_keep_grad_for_cuda_graph", True)
                    if dist_param.grad is not None:
                        del dist_param.grad
                else:
                    assert dist_grad is None or dist_param.dtype == dist_grad.dtype, (
                        f"{name} Dist param dtype {dist_param.dtype} does not match "
                        f"dist grad dtype {dist_grad.dtype}"
                    )
                    setattr(dist_param, "grad", dist_grad)
                    if param_group.enable_full_iteration_cuda_graph and dist_grad is not None:
                        setattr(dist_param, "_mfsdp_keep_grad_for_cuda_graph", True)
                    if hasattr(dist_param, "decoupled_grad"):
                        dist_param.decoupled_grad = None

            if async_op:
                event = completion_stream.record_event()
                ctx.reduce_grad_buckets[id(self)].append((event, param_group))

            # NaN check after reduction
            if getattr(self, "_enable_nan_checks", False):
                for name, dist_grad in zip(param_names, param_group.optimizer_grads):
                    if dist_grad is not None:
                        assert not torch.isnan(
                            dist_grad._local_tensor
                        ).any(), f"NaN in dist grad for parameter {name}"

        torch.cuda.nvtx.range_pop()

    @torch.no_grad()
    def finish_grad_sync(self, force_all_reduce: Optional[bool] = False):
        """Finish optimizer-facing gradient synchronization for this iteration."""
        assert not force_all_reduce, "FSDP v2 does not support force_all_reduce."
        caller_stream = torch.cuda.current_stream()
        for stream in self._fsdp_root_context.rs_streams:
            caller_stream.wait_stream(stream)

    @torch.no_grad()
    def _scale_gradients(self, scaling_factor: float):
        """Scale optimizer-facing gradients by a factor."""
        ctx = self._fsdp_root_context
        caller_stream = torch.cuda.current_stream()
        for stream in ctx.rs_streams:
            caller_stream.wait_stream(stream)
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                for dist_param in param_group.optimizer_params:
                    grad = getattr(dist_param, "decoupled_grad", None)
                    if grad is None:
                        grad = dist_param.grad
                    if grad is None:
                        continue
                    if isinstance(grad, DTensor):
                        grad._local_tensor.mul_(scaling_factor)
                    else:
                        grad.mul_(scaling_factor)

    def zero_grad(self, set_to_none: bool = True):
        """Zero gradients for all parameter groups."""
        for child in self._get_fsdp_modules(recursive=True):
            for param_group in child._fsdp_param_groups:
                param_group.zero_grad(set_to_none=set_to_none)

    def _release_grad_storage_if_unused(self) -> None:
        """Release stale gradient storage across the complete FSDP root.

        A plain ``torch.optim.Optimizer.zero_grad(set_to_none=True)`` clears
        optimizer-facing ``dist_param.grad`` references without calling
        :meth:`FSDPModule.zero_grad`, so the parameter-group accumulation flags
        may still describe the previous step. At the next root forward, an
        absence of optimizer-facing gradients across *all* parameter groups is
        an optimizer-boundary signal. Reset those stale flags through the
        parameter-group zero-grad path and release every eligible grad buffer
        before any parameter unshard can overlap it.

        If any gradient is still live, the model may be between accumulated
        microbatches, so this method leaves every group untouched.

        Full-iteration CUDA graph mode owns stable optimizer-facing gradient
        storage across iterations. Its zeroing is part of the graph-compatible
        optimizer lifecycle, so this eager root sweep must not inspect or
        mutate parameter-group gradient state in that mode.
        """
        if self._fsdp_state.enable_full_iteration_cuda_graph:
            return

        param_groups = [
            param_group
            for child in self._get_fsdp_modules(recursive=True)
            for param_group in child._fsdp_param_groups
        ]
        if any(
            getattr(dist_param, "grad", None) is not None
            or getattr(dist_param, "decoupled_grad", None) is not None
            for param_group in param_groups
            for dist_param in param_group.optimizer_params
        ):
            return

        for param_group in param_groups:
            param_group.zero_grad(set_to_none=True)

    def _zero_grad_buffer(self):
        """Zero the gradient buffer for all parameter groups."""
        self.zero_grad(set_to_none=False)

    def _copy_main_weights_to_model_weights(self):
        """Copy main weight buffer to model weight buffer."""
        for child in self.modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_group in child._fsdp_param_groups:
                param_group.refresh_model_weight()
        # Explicit optimizer integrations and the lazy pre-forward path share
        # this method. Whichever installs the weights first consumes the work.
        self._fsdp_root_context.model_weight_refresh_pending = False

    def _log_parameter_groups(self):
        """Print a compact summary of rewrite-path FSDP parameter groups."""

        def _fmt_dtype(dtype: torch.dtype) -> str:
            short = {
                torch.float32: "fp32",
                torch.float16: "fp16",
                torch.bfloat16: "bf16",
                torch.int64: "i64",
                torch.int32: "i32",
                torch.uint8: "u8",
            }
            return short.get(dtype, str(dtype).removeprefix("torch."))

        def _elem_size(dtype: torch.dtype) -> int:
            return {
                torch.float32: 4,
                torch.float16: 2,
                torch.bfloat16: 2,
                torch.int64: 8,
                torch.int32: 4,
                torch.uint8: 1,
            }.get(dtype, 1)

        def _mb(num_bytes: int | float) -> str:
            return f"{num_bytes / 1_000_000:.2f} MB"

        rank = torch.distributed.get_rank()
        lines = [f"FSDP parameter groups (rank {rank})"]
        group_idx = 0
        total_model_elems = 0
        total_comm = 0
        total_pad = 0

        for module_name, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            for param_names, param_group in child._named_param_groups:
                param_shapes = [p.shape for p in param_group.params]
                numel = sum(s.numel() for s in param_shapes)
                total_model_elems += numel
                dp_size = param_group.mesh.size(-1)

                buffer_entries = []
                group_pad = 0
                group_comm = 0
                buffer_metadata, model_weight_ranges = param_group.buffer_diagnostics()
                for (
                    buffer_label,
                    buffer_dtype,
                    data_size,
                    global_size,
                    outer_sharded,
                    inner_sharded,
                ) in buffer_metadata:
                    elem_size = _elem_size(buffer_dtype)
                    group_pad += max(0, global_size - numel) * elem_size
                    group_comm += global_size * elem_size
                    dist_flag = "O" if outer_sharded else "I" if inner_sharded else "R"
                    buffer_entries.append(
                        f"{buffer_label}[{_fmt_dtype(buffer_dtype)}:{data_size}:{dist_flag}]"
                    )
                total_pad += group_pad
                total_comm += group_comm

                lines.append(
                    f"- {module_name} #{group_idx} dp={dp_size} "
                    f"layout={param_group.layout} "
                    f"chunk_factor={param_group.chunk_size_factor}"
                )
                lines.append(
                    f"  {numel:,} elems x {_fmt_dtype(param_group.dtype)} "
                    f"comm={_mb(group_comm)} pad={_mb(group_pad)} "
                    f"{' '.join(buffer_entries)}"
                )
                for param_name, param_shape, model_weight_range in zip(
                    param_names, param_shapes, model_weight_ranges
                ):
                    offset_info = ""
                    if model_weight_range is not None:
                        offset, size = model_weight_range
                        offset_info = f" @{offset:,}+{size:,}"
                    lines.append(f"    {param_name:50s} {str(tuple(param_shape)):24s}{offset_info}")
                group_idx += 1

        lines.append(
            f"Summary: {group_idx} groups, {total_model_elems:,} model elems, "
            f"comm={_mb(total_comm)}, pad={_mb(total_pad)}"
        )
        logger.info("\n".join(lines))

    def _set_nan_check(self, enable_nan_checks: bool):
        """Enable or disable NaN checking."""
        for _, child in self.named_modules():
            if not isinstance(child, FSDPModule):
                continue
            setattr(child, "_enable_nan_checks", enable_nan_checks)

        if enable_nan_checks:
            for name, param in self.named_parameters():
                if isinstance(param, DTensor):
                    param_data = param.data._local_tensor
                else:
                    param_data = param.data
                assert not torch.isnan(param_data).any(), f"NaN detected in parameter {name}"
            for child in self.modules():
                if not isinstance(child, FSDPModule):
                    continue
                for param_group in child._fsdp_param_groups:
                    param_group.assert_model_weights_not_nan()

    def get_root_module(self):
        """Return the root FSDP module associated with this module."""
        return self._fsdp_root_context.get_root_module()

    def set_is_last_backward(self, is_last_backward: bool = True):
        """Set whether the next backward is the optimizer-step boundary.

        This mirrors PyTorch FSDP2's microbatching API.  On the last backward,
        delayed inner grad reductions and outer-DP grad sync are issued.
        """
        self._fsdp_root_context.is_last_backward = is_last_backward

    @contextmanager
    def no_sync(self):
        """Defer the outer-DP / HSDP gradient reduce until the last micro-batch
        (like MegatronFSDP v1 / PyTorch DDP ``no_sync``).

        Example::

            with model.no_sync():
                loss(mb0).backward()   # accumulate, no reduce
            loss(mb1).backward()       # last micro-batch -> reduce fires
        """
        if not self._fsdp_state._is_root:
            yield
            return
        self.set_is_last_backward(False)
        try:
            yield
        finally:
            self.set_is_last_backward(True)

    def _sync_module_states_after_load(self):
        self._copy_main_weights_to_model_weights()


def _get_module_fsdp_param_groups(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy,
    mesh: Optional[DeviceMesh] = None,
    ignored_params: Optional[set[nn.Parameter]] = None,
    gradient_scaling_factor: Optional[float] = None,
    sharding_strategy: str = "optim_grads_params",
    outer_dp_sharding_strategy: str = "no_shard",
) -> List[ParameterGroup]:
    """
    Group module parameters by (device, dtype, requires_grad) and create ParameterGroups.

    Parameters are grouped because they share the same buffer management
    and sharding strategy. Each group gets its own DataParallelBuffer.
    """
    param_groups = {}

    for param in module.parameters():
        if ignored_params is not None and param in ignored_params:
            continue

        # The policy owns dtype-sensitive grouping.
        param_dtype = mp_policy.group_key_dtype(param)
        param_attrs = (param.device, param_dtype, param.requires_grad)
        if param_attrs not in param_groups:
            param_groups[param_attrs] = []
        param_groups[param_attrs].append(param)

    # Create ParameterGroup for each group
    fsdp_param_groups = []
    for i, params in enumerate(param_groups.values()):
        if mesh is None:
            raise ValueError("ParameterGroup requires an explicit DeviceMesh")
        layout = ParameterGroupLayout.from_strategies(
            sharding_strategy, outer_dp_sharding_strategy if mesh.ndim == 2 else None
        )
        param_group = ParameterGroup(
            params,
            mesh=mesh,
            param_group_id=ParamGroupIdx(id(module), i),
            layout=layout,
            mp_policy=mp_policy,
            gradient_scaling_factor=gradient_scaling_factor,
        )
        fsdp_param_groups.append(param_group)

    return fsdp_param_groups
