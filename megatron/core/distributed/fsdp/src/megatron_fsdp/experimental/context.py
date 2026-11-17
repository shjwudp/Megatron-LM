# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared runtime state for Megatron-FSDP modules."""

import enum
from collections.abc import Callable
from weakref import WeakKeyDictionary

import torch
from torch import nn
from torch.distributed import ProcessGroup

from .execution_order import ExecutionAction, ExecutionEvent, ExecutionOrder, ExecutionOrderView
from .gradient_window import GradientWindow
from .phase import Phase
from .runtime_policy import RuntimePolicy


class PrefetchOrder(enum.Enum):
    """How to choose the module order used for prefetching."""

    REGISTRATION = enum.auto()
    RECORDED = enum.auto()


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    inner_reduction_stream: torch.cuda.Stream
    # Aliases inner_reduction_stream unless DP-outer overlap is enabled.
    outer_reduction_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because each parameter group tracks whether model_weight is stale
    # after syncing from main_weight.
    is_last_microbatch: bool
    use_symmetric_memory: bool
    caller_managed_grad_sync: bool
    # Prefetch reads only unshard requests from the shared source.
    execution_order: ExecutionOrder
    # Topology metadata must not keep modules alive after construction.
    _roots: WeakKeyDictionary[nn.Module, None]
    # Names are relative to each FSDP root and absent until finalization.
    _module_names: WeakKeyDictionary[nn.Module, str]
    # The optimizer runs on the current stream and must wait for reductions on
    # this context's reduce-scatter stream. Each context owns its own stream, so
    # independent roots sharing a context need only one completion callback.
    _post_backward_hook_registered: bool

    def __init__(
        self,
        device: torch.device | None = None,
        runtime_policy: RuntimePolicy = RuntimePolicy(),
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
        prefetch_order: PrefetchOrder = PrefetchOrder.REGISTRATION,
        prefetch_process_group: ProcessGroup | None = None,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: CUDA device on which this context schedules communication. Defaults to
                the current CUDA device.
            runtime_policy: Communication allocation, stream, and gradient-synchronization
                settings shared by all modules constructed in this context.
            parameter_to_owner: Construction-time owner assignments for TensorAtomic
                parameters, keyed by the original parameters before sharding. Owners are
                ranks in each parameter group's 1-D data-parallel mesh and must agree across
                that mesh. Every TensorAtomic parameter needs an entry; other entries are
                ignored. Tensors are packed by owner without changing logical parameter order.
            prefetch_order: ``REGISTRATION`` expects a forward/backward pass matching module
                registration order. ``RECORDED`` gathers parameters on demand and records
                module calls until ``finish_grad_sync()``. This runs automatically after
                the last microbatch unless gradient synchronization is caller-managed.
                Subsequent passes must repeat the complete sequence consistently across
                data-parallel ranks.
                Recording currently requires the ordinary module hooks; combined 1F1B
                bypasses them. Discard the context if the recorded pass raises.
            prefetch_process_group: Explicit agreement group whose ranks share the same
                gradient contribution trace. Required only for configured gradient windows.
        """
        overlap_allgather_reduction = runtime_policy.overlap_allgather_reduction
        overlap_outer_communication = runtime_policy.overlap_outer_communication
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise ValueError(f"FsdpContext requires a CUDA device, got {device}.")

        self.is_last_microbatch = True
        self.use_symmetric_memory = runtime_policy.use_symmetric_memory
        self.caller_managed_grad_sync = runtime_policy.caller_managed_grad_sync
        self.prefetch_order = prefetch_order
        self._prefetch_process_group = prefetch_process_group
        self._gradient_window: GradientWindow | None = None
        self.execution_order = ExecutionOrder()
        self._prefetch_order_view: ExecutionOrderView | None = None
        self._recording_execution_order = prefetch_order is PrefetchOrder.RECORDED
        self._post_backward_hook_registered = False
        self._delayed_post_backward_callbacks: list[
            tuple[nn.Module, Callable[[nn.Module], None]]
        ] = []
        # Construction-only; empty after finalization.
        self._registered_modules: list[nn.Module] = []
        self._roots = WeakKeyDictionary()
        self._module_names = WeakKeyDictionary()
        self.parameter_to_owner = parameter_to_owner
        self._is_finalized = False
        # The two overlap options independently split streams from the inner
        # reduction stream. With (overlap_allgather_reduction,
        # overlap_outer_communication), the assignments are:
        #   (False, False): all-gather and both reductions share one stream.
        #   (True, False):  all-gather is separate; both reductions share a stream.
        #   (False, True):  all-gather and inner reduction share; outer is separate.
        #   (True, True):   all-gather, inner reduction, and outer reduction each
        #                  have a separate stream.
        # Outer all-gather separation is not implemented yet; the outer option
        # currently affects only reduction.
        self.inner_reduction_stream = torch.cuda.Stream(device)
        self.allgather_stream = (
            torch.cuda.Stream(device)
            if overlap_allgather_reduction
            else self.inner_reduction_stream
        )
        self.outer_reduction_stream = (
            torch.cuda.Stream(device)
            if overlap_outer_communication
            else self.inner_reduction_stream
        )

    @property
    def is_finalized(self) -> bool:
        """Whether construction has completed for this context."""
        return self._is_finalized

    @property
    def is_recording_execution_order(self) -> bool:
        """Whether module calls are being recorded for subsequent prefetching."""
        return self._recording_execution_order

    @property
    def prefetch_order_view(self) -> ExecutionOrderView:
        """Unshard requests and cursor used by prefetch.

        Other actions in the shared source do not advance this view's cursor.
        Replacing ``execution_order`` creates a fresh view and cursor.
        """
        view = self._prefetch_order_view
        if view is None or view.source is not self.execution_order:
            view = self.execution_order.view(action=ExecutionAction.UNSHARD)
            self._prefetch_order_view = view
        return view

    def register_module(self, module: nn.Module) -> None:
        """Register a module after its FSDP initialization completes in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots, names, and cross-root prefetch orders."""
        if self._is_finalized:
            raise RuntimeError("FSDP context is already finalized.")

        # Successful FsdpModule initialization registers the module, and fully_shard
        # rejects children from another context. Membership identifies this context's
        # FSDP modules without importing FsdpModule.
        registered_modules = set(self._registered_modules)
        children: set[nn.Module] = set()
        for module in self._registered_modules:
            _collect_fsdp_children(module, registered_modules, children)
        for root in self._registered_modules:
            if root in children:
                continue
            self._roots[root] = None
            for name, module in root.named_modules():
                if module not in registered_modules:
                    continue
                self._module_names[module] = name
                if not self._recording_execution_order:
                    self.execution_order.append(ExecutionEvent(module, Phase.FORWARD))

        if not self._recording_execution_order:
            for root in reversed(list(self._roots)):
                _collect_backward_order(root, registered_modules, self.execution_order)

        self._registered_modules.clear()
        self.parameter_to_owner = None
        self._is_finalized = True

    def is_root(self, module: nn.Module) -> bool:
        """Return whether ``module`` is an outermost FSDP module in this context."""
        self.ensure_finalized()
        return module in self._roots

    def module_name(self, module: nn.Module) -> str:
        """Return ``module``'s name relative to its FSDP root."""
        self.ensure_finalized()
        return self._module_names[module]

    def ensure_finalized(self) -> None:
        """Raise if construction has not completed for this context."""
        if not self._is_finalized:
            raise RuntimeError(
                "FSDP context is not finalized. Exit fully_shard_context to complete construction."
            )

    def current_stream(self) -> torch.cuda.Stream:
        """Current stream on this context's device."""
        return torch.cuda.current_stream(self.allgather_stream.device)

    def validate_grad_sync(self) -> None:
        """Require a caller-owned wait or a pending autograd completion callback."""
        if self.caller_managed_grad_sync or self._post_backward_hook_registered:
            return
        raise RuntimeError(
            "Gradient reduction has no pending autograd completion callback. "
            "Use FsdpContext(runtime_policy=RuntimePolicy(caller_managed_grad_sync=True)) "
            "and call context.finish_grad_sync() after all backward work, including "
            "delayed weight-gradient computation, before consuming gradients."
        )

    def finish_grad_sync(self) -> None:
        """Submit gradient-window tails and wait on submitted reductions.

        Call after all backward work, including delayed weight-gradient computation,
        and before reading or modifying gradients. Stream waits enqueue dependencies
        without blocking the CPU. The first recorded gradient window also performs
        rank agreement and reads its result on the CPU before compiling the policy.
        """
        if self._gradient_window is not None:
            if self._delayed_post_backward_callbacks:
                raise RuntimeError("Finish delayed backward callbacks before gradient-window sync.")
            self._gradient_window.finish_step()
        self.current_stream().wait_stream(self.inner_reduction_stream)
        if self.outer_reduction_stream is not self.inner_reduction_stream:
            self.current_stream().wait_stream(self.outer_reduction_stream)
        self._recording_execution_order = False

    def configure_gradient_window(
        self, *, window_size: int, budget_bytes: int, eligible_modules
    ) -> None:
        """Bound local accumulation before inner reduction, configured before compute.

        The byte budget includes main-gradient-dtype accumulators, communication
        casts and submitted inputs until retirement. Persistent gradients and
        ineligible groups' baseline staging are outside this budget.
        """
        self.ensure_finalized()
        if any(not module._schedule_policy.register_hooks for module in self._module_names):
            raise ValueError("Gradient windows require ordinary FSDP module hooks.")
        if self._gradient_window is not None:
            raise RuntimeError("FSDP gradient windows have already been configured.")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Configure gradient windows before graph capture.")
        if not self.caller_managed_grad_sync or self._prefetch_process_group is None:
            raise ValueError(
                "Gradient windows require caller-managed sync and an explicit agreement group."
            )
        if window_size > 1 and self.use_symmetric_memory:
            raise ValueError("Bounded gradient windows do not support symmetric-memory staging.")
        if self._delayed_post_backward_callbacks or any(
            group._unshard_events
            for module in self._module_names
            for group in module.parameter_groups
        ):
            raise RuntimeError("Configure gradient windows at a clean synchronization boundary.")
        self._gradient_window = GradientWindow(
            self, self._module_names, eligible_modules, window_size, budget_bytes
        )

    def prepare_for_cuda_graph_capture(self) -> None:
        """Pin the completed eager gradient trace before entering graph capture."""
        if self._gradient_window is None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare gradient windows before graph capture.")
        if self._delayed_post_backward_callbacks or self.is_recording_execution_order:
            raise RuntimeError(
                "Gradient capture requires a complete eager synchronization boundary."
            )
        self._gradient_window.prepare_capture()

    def accumulate_gradient_window(self, module, group, *, group_index: int) -> bool:
        """Consume one completed local contribution, or select the unchanged baseline path."""
        return self._gradient_window is not None and self._gradient_window.contribute(
            module, group, group_index
        )

    def gradient_window_reason(self, group) -> str:
        return "baseline" if self._gradient_window is None else self._gradient_window.reason(group)

    def _flush_inner_gradient_window(self, group, *, reason: str, partial_grad=None) -> None:
        """The actual inner reduction boundary for baseline and merged contributions."""
        if partial_grad is not None:
            group.reduce_inner_gradients(partial_grad)
            if self._gradient_window is not None:
                self._gradient_window.legacy_flush(reason=reason)
        else:
            if self._gradient_window is None:
                raise RuntimeError("Cannot flush an unconfigured gradient window.")
            self._gradient_window.flush(group, reason=reason)

    def _finalize_gradient_outer(self, group) -> None:
        """Finalize the original DP-outer stage after this group's inner tail."""
        if self.outer_reduction_stream is not self.inner_reduction_stream:
            self.outer_reduction_stream.wait_stream(self.inner_reduction_stream)
        with torch.cuda.stream(self.outer_reduction_stream):
            group.reduce_outer_gradients()
        if self._gradient_window is not None:
            self._gradient_window.outer_finished(group)

    def validate_gradient_window_capture(self) -> None:
        """Validate policy before replay; captured sharded gradient views must remain installed."""
        policy = self._gradient_window
        if policy is None:
            return
        policy.validate_capture()
        if any(not group._has_sharded_grads() for group in policy.seen_groups):
            raise RuntimeError("Captured gradient windows require zero_grad(set_to_none=False).")

    def post_backward(self) -> None:
        """Finish backward cleanup and automatically synchronize the last microbatch."""
        self._run_delayed_post_backward_callbacks()
        # Standalone training can call optimizer.step() immediately after backward.
        # Wait for reductions on the last microbatch; earlier microbatches keep
        # overlapping reduction with compute. MCore calls finish_grad_sync() itself
        # after its schedule completes, including delayed weight gradients.
        if not self.caller_managed_grad_sync and self.is_last_microbatch:
            self.finish_grad_sync()
        self._post_backward_hook_registered = False

    def _run_delayed_post_backward_callbacks(self) -> None:
        """Run deferred module completion callbacks once."""
        # Finish children before parents to close nested backward NVTX ranges.
        while self._delayed_post_backward_callbacks:
            module, callback = self._delayed_post_backward_callbacks.pop()
            callback(module)

    def register_post_backward_hook(self) -> None:
        """Register one final callback for deferred cleanup and optional gradient sync.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self._post_backward_hook_registered:
            # Another root sharing this context already queued the completion wait.
            return
        self._post_backward_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self.post_backward)


def _collect_backward_order(
    module: nn.Module, registered_modules: set[nn.Module], order: ExecutionOrder
) -> None:
    """Collect one root's static backward prefetch order."""
    if module in registered_modules:
        order.append(ExecutionEvent(module, Phase.BACKWARD))

    for child in reversed(list(module.children())):
        _collect_backward_order(child, registered_modules, order)


def _collect_fsdp_children(
    module: nn.Module, registered_modules: set[nn.Module], children: set[nn.Module]
) -> None:
    """Collect the nearest registered FSDP descendants of ``module``."""
    for child in module.children():
        if child in registered_modules:
            children.add(child)
        else:
            _collect_fsdp_children(child, registered_modules, children)
