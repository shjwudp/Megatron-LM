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

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from weakref import WeakKeyDictionary

import torch
from torch import nn

from .execution_order import ExecutionOrder

_FSDP_CONTEXT = ContextVar["FsdpContext | None"]("mfsdp_context", default=None)


class FsdpContext:
    """Runtime stream and prefetch state shared by FSDP roots constructed together."""

    allgather_stream: torch.cuda.Stream
    reduce_scatter_stream: torch.cuda.Stream
    # HFSDP/HSDP need explicit last-microbatch state. First-microbatch state is
    # unnecessary because each parameter group tracks whether model_weight is stale
    # after syncing from main_weight.
    is_last_microbatch: bool
    use_symmetric_memory: bool
    unify_communication_stream: bool
    caller_managed_grad_sync: bool
    """Leave gradient synchronization to the caller instead of an autograd callback.

    The caller must call ``finish_grad_sync()`` after all gradient producers,
    including delayed weight-gradient computation, and before consuming gradients.
    """
    # Unavailable until an iteration is recorded. These refer to the active phase's trace.
    forward_order: ExecutionOrder[nn.Module] | None
    backward_order: ExecutionOrder[nn.Module] | None
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
        use_symmetric_memory: bool = False,
        unify_communication_stream: bool = False,
        parameter_to_owner: dict[nn.Parameter, int] | None = None,
        caller_managed_grad_sync: bool = False,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: CUDA device on which this context schedules communication. Defaults to
                the current CUDA device.
            use_symmetric_memory: Whether modules constructed in this context allocate
                communication staging buffers from PyTorch's NCCL symmetric-memory pool.
            unify_communication_stream: Whether all-gathers and reduce-scatters share one
                communication stream to reduce peak transient memory. See
                https://github.com/NVIDIA/Megatron-LM/issues/6471.
            parameter_to_owner: Construction-time owner assignments for TensorAtomic
                parameters, keyed by the original parameters before sharding. Owners are
                ranks in each parameter group's 1-D data-parallel mesh and must agree across
                that mesh. Every TensorAtomic parameter needs an entry; other entries are
                ignored. Tensors are packed by owner without changing logical parameter order.
            caller_managed_grad_sync: Disable the automatic autograd completion callback,
                allowing delayed weight gradients or custom backward schedules. The caller must
                call ``finish_grad_sync()`` after all backward work and before reading or
                modifying gradients.
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type != "cuda":
            raise ValueError(
                f"fully_shard_context/FsdpContext requires a CUDA device, got {device}."
            )

        self.is_last_microbatch = True
        self.use_symmetric_memory = use_symmetric_memory
        self.unify_communication_stream = unify_communication_stream
        self.caller_managed_grad_sync = caller_managed_grad_sync
        self.forward_order = None
        self.backward_order = None
        # Keep only the latest training and forward-only schedule patterns.
        self._iteration_traces: dict[
            bool, tuple[object, ExecutionOrder[nn.Module], ExecutionOrder[nn.Module] | None]
        ] = {}
        self._iteration_active = False
        self._recording = False
        self._post_backward_hook_registered = False
        # Construction-only; empty after finalization.
        self._registered_modules: list[nn.Module] = []
        self._roots = WeakKeyDictionary()
        self._module_names = WeakKeyDictionary()
        self.parameter_to_owner = parameter_to_owner
        self._is_finalized = False
        self._context_token: Token[FsdpContext | None] | None = None
        self.allgather_stream = torch.cuda.Stream(device)
        if unify_communication_stream:
            # A unified stream lets an all-gather reuse the storage released by a
            # preceding reduce-scatter.
            self.reduce_scatter_stream = self.allgather_stream
        else:
            self.reduce_scatter_stream = torch.cuda.Stream(device)

    def __enter__(self) -> "FsdpContext":
        """Activate this context for FSDP module construction."""
        if _FSDP_CONTEXT.get() is not None:
            raise RuntimeError("fully_shard_context does not support nesting.")
        if self._is_finalized:
            raise RuntimeError("Cannot enter fully_shard_context after construction is finalized.")
        self._context_token = _FSDP_CONTEXT.set(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: object) -> None:
        """Finalize successful construction and always clear the active scope."""
        assert self._context_token is not None
        try:
            if exc_type is None:
                self.finalize()
        finally:
            _FSDP_CONTEXT.reset(self._context_token)
            self._context_token = None

    def register_module(self, module: nn.Module) -> None:
        """Register a module after its FSDP initialization completes in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots and names; runtime execution orders are recorded later."""
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

        self._registered_modules.clear()
        self.parameter_to_owner = None
        self._is_finalized = True

    @contextmanager
    def iteration(self, *, key: object = None, forward_only: bool = False) -> Iterator[None]:
        """Record or replay a complete schedule, including every VPP chunk and microbatch.

        A new pattern records demand calls with speculative prefetch disabled.
        Successful recording installs the trace; replay resets both cursors once
        at entry and requires complete consumption at exit. Forward-only calls
        have their own trace and never replace the training trace. Change ``key``
        when the schedule changes. All ranks must follow collective-compatible,
        repeatable patterns. Outside this scope, parameters are loaded on demand.

        Failed scopes do not install a new trace. A failed forward/backward may
        still leave model state or collectives incomplete; discard that work.
        This Python-hook prototype requires CUDA graphs to be disabled.
        """
        self.ensure_finalized()
        if self._iteration_active:
            raise RuntimeError("An FSDP iteration is already active.")
        previous = self.forward_order, self.backward_order
        cached = self._iteration_traces.get(forward_only)
        recording = cached is None or cached[0] != key
        if recording:
            forward_order = ExecutionOrder()
            backward_order = None if forward_only else ExecutionOrder()
        else:
            _, forward_order, backward_order = cached
            forward_order.reset()
            if backward_order is not None:
                backward_order.reset()
        self.forward_order, self.backward_order = forward_order, backward_order
        self._iteration_active = True
        self._recording = recording
        try:
            yield
            orders = [forward_order]
            if backward_order is not None:
                orders.append(backward_order)
            if recording:
                if any(not list(order) for order in orders):
                    raise RuntimeError("An FSDP recording must observe every requested phase.")
                self._iteration_traces[forward_only] = (key, forward_order, backward_order)
            elif any(not order.is_complete() for order in orders):
                raise RuntimeError("FSDP iteration ended before its execution order was consumed.")
        except BaseException:
            self.forward_order, self.backward_order = previous
            raise
        finally:
            self._iteration_active = False
            self._recording = False

    def execution_order(
        self, module: nn.Module, *, backward: bool = False
    ) -> ExecutionOrder[nn.Module] | None:
        """Observe a demand call and return its ready-to-prefetch order, if any."""
        if not self._iteration_active:
            return None
        order = self.backward_order if backward else self.forward_order
        if order is None:
            raise RuntimeError("Backward calls are not allowed in a forward-only iteration.")
        if self._recording:
            order.append(module)
            return None
        # Validate the next occurrence; the sequence itself never looks up a module.
        if order.next_item() is not module:
            raise RuntimeError("FSDP module calls diverged from the recorded execution order.")
        order.advance()
        return order

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
            "Use fully_shard_context(caller_managed_grad_sync=True) and call "
            "context.finish_grad_sync() after all backward work, including "
            "delayed weight-gradient computation, before consuming gradients."
        )

    def finish_grad_sync(self) -> None:
        """Order current-stream consumers after all gradient reductions submitted so far.

        Call after all backward work, including delayed weight-gradient computation,
        and before reading or modifying gradients. This enqueues a stream dependency;
        it does not block the CPU or wait for reductions that have not yet been launched.
        """
        self.current_stream().wait_stream(self.reduce_scatter_stream)

    def post_backward(self) -> None:
        """Order current-stream consumers after this backward's gradient reductions."""
        self.finish_grad_sync()
        self._post_backward_hook_registered = False

    def register_post_backward_hook(self) -> None:
        """Register one final callback unless the caller manages gradient synchronization.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self.caller_managed_grad_sync:
            # Leave the wait to the caller's finish_grad_sync(), after all backward work,
            # including delayed weight-gradient computation, has been launched.
            return
        if self._post_backward_hook_registered:
            # Another root sharing this context already queued the completion wait.
            return
        self._post_backward_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self.post_backward)


def current_fully_shard_context() -> FsdpContext | None:
    """Return the innermost active ``fully_shard_context``, or ``None``.

    Read-only counterpart of :func:`fully_shard_context`: it never creates, joins, or
    finalizes a context, and returns ``None`` whenever no ``fully_shard_context`` scope is
    active. Callers that must share one context -- for example per-chunk wrappers built by
    a single wrap call -- use it to join the caller's ambient context instead of opening a
    second one.
    """
    return _FSDP_CONTEXT.get()


def _collect_fsdp_children(
    module: nn.Module, registered_modules: set[nn.Module], children: set[nn.Module]
) -> None:
    """Collect the nearest registered FSDP descendants of ``module``."""
    for child in module.children():
        if child in registered_modules:
            children.add(child)
        else:
            _collect_fsdp_children(child, registered_modules, children)
