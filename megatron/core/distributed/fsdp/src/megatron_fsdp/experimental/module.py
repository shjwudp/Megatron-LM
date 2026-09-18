# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Module mixin for the minimal Megatron-FSDP path."""

import enum
import logging
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager
from typing import Literal, cast
from weakref import ref

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import MixedPrecisionPolicy, fp8_need_transpose_data, is_float8tensor
from ..utils import log_single_rank
from .countdown import MultiplicityReadiness
from .indexed_order import IndexedOrder
from .module_utils import get_parameter_owner
from .parameter_group import (
    Fp8ParameterGroup,
    FsdpParameter,
    FsdpParameterGroup,
    get_containing_parameter_group,
)
from .placement import Flat
from .quantization import (
    BOTH,
    COLWISE,
    ROWWISE,
    PayloadOrientation,
    merge_orientations,
    orientation_directions,
)
from .schedule import SchedulePolicy

logger = logging.getLogger(__name__)


def _is_in_backward() -> bool:
    """Return whether the current thread is executing an autograd GraphTask."""
    return torch._C._current_graph_task_id() != -1


def _is_fp8_parameter(parameter: nn.Parameter) -> bool:
    """Whether ``parameter`` is a TE MXFP8 primary weight with two payloads.

    Such a parameter rests as separate row-wise (forward GEMM) and column-wise
    (backward GEMM) payload buffers, so which of them an unshard must gather
    depends on the pass.
    """
    return is_float8tensor(parameter) and fp8_need_transpose_data(parameter)


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
    # Static orders used to drive all-gather prefetch. We may want to switch to
    # capturing runtime order if static module order proves too fragile. Each
    # FsdpModule tracks its own materialized state via ``FsdpModule._unshard_event``.
    forward_order: IndexedOrder["FsdpModule"]
    backward_order: IndexedOrder["FsdpModule"]
    # The optimizer runs on the current stream and must wait for reductions on
    # this context's reduce-scatter stream. Each context owns its own stream, so
    # independent roots sharing a context need only one completion callback.
    _post_backward_hook_registered: bool

    def __init__(
        self,
        device: torch.device,
        use_symmetric_memory: bool = False,
        unify_communication_stream: bool = False,
    ) -> None:
        """Create rank-local runtime state for FSDP modules on ``device``.

        Args:
            device: Device on which this context schedules communication.
            use_symmetric_memory: Whether modules constructed in this context allocate
                communication staging buffers from PyTorch's NCCL symmetric-memory pool.
            unify_communication_stream: Whether all-gathers and reduce-scatters share one
                communication stream to reduce peak transient memory.
        """
        self.is_last_microbatch = True
        self.use_symmetric_memory = use_symmetric_memory
        self.unify_communication_stream = unify_communication_stream
        self.forward_order = IndexedOrder()
        self.backward_order = IndexedOrder()
        self._post_backward_hook_registered = False
        # Construction-only; empty after finalization.
        self._registered_modules: list[FsdpModule] = []
        self._is_finalized = False
        self.allgather_stream = torch.cuda.Stream(device)
        if unify_communication_stream:
            # A unified stream lets an all-gather reuse the storage released by a
            # preceding reduce-scatter.
            self.reduce_scatter_stream = self.allgather_stream
        else:
            self.reduce_scatter_stream = torch.cuda.Stream(device)

    def register_module(self, module: "FsdpModule") -> None:
        """Register a module constructed in this context."""
        if self._is_finalized:
            raise RuntimeError("Cannot register an FSDP module after its context is finalized.")
        self._registered_modules.append(module)

    def finalize(self) -> None:
        """Finalize roots, names, and cross-root prefetch orders."""
        if self._is_finalized:
            raise RuntimeError("FSDP context is already finalized.")

        children: set[FsdpModule] = set()
        for module in self._registered_modules:
            _collect_fsdp_children(cast(nn.Module, module), children)
        # FsdpModules that are not descendants of any other FsdpModule.
        roots = [module for module in self._registered_modules if module not in children]

        for root in roots:
            root._is_root = True
            for name, module in cast(nn.Module, root).named_modules():
                if not isinstance(module, FsdpModule):
                    continue
                module._name = name
                self.forward_order.append(module)

        for root in reversed(roots):
            _collect_backward_order(cast(nn.Module, root), self.backward_order)

        self._registered_modules.clear()
        self._is_finalized = True

    def ensure_finalized(self) -> None:
        """Raise if construction has not completed for this context."""
        if not self._is_finalized:
            raise RuntimeError(
                "FSDP context is not finalized. Exit fully_shard_context before running forward."
            )

    def current_stream(self) -> torch.cuda.Stream:
        """Current stream on this context's device."""
        return torch.cuda.current_stream(self.allgather_stream.device)

    def post_backward(self) -> None:
        """Order current-stream consumers after this context's gradient reductions."""
        self.current_stream().wait_stream(self.reduce_scatter_stream)
        self._post_backward_hook_registered = False

    def register_post_backward_hook(self) -> None:
        """Register one context-level final callback for the current backward.

        Multiple FSDP roots can share this context. Waiting for the
        reduce-scatter stream in each root's ``post_backward()`` would prevent
        one root's backward compute from overlapping another root's gradient
        reductions. Wait once at context-level autograd completion instead.
        """

        if self._post_backward_hook_registered:
            return
        self._post_backward_hook_registered = True

        # TODO(wujingyue): Switch to torch.autograd.graph.queue_callback() when Megatron-LM
        # requires a PyTorch version that includes it:
        # https://github.com/pytorch/pytorch/pull/193958
        torch.autograd.Variable._execution_engine.queue_callback(self.post_backward)


class FsdpModule:
    """Mixin attached to modules managed by the minimal FSDP path."""

    class Phase(enum.Enum):
        """Lifecycle phase of this FsdpModule."""

        RESTING = enum.auto()
        FORWARD = enum.auto()
        BACKWARD = enum.auto()

    # Name relative to the root FSDP module from named_modules().
    # Root uses "" and None means uninitialized.
    _name: str | None
    _parameter_groups: tuple[FsdpParameterGroup, ...]
    _context: FsdpContext
    # Exact per-parameter gradient accounting against a declared multiplicity.
    # The automatic path declares 1 for every trainable parameter; the
    # combined/fine-grained 1F1B path declares the real counts.
    _grad_readiness: MultiplicityReadiness | None
    # Per-parameter expected contribution counts declared by the schedule, keyed the
    # same way as the marks. Empty means "every trainable parameter contributes
    # once", which is the automatic path.
    _grad_multiplicity: dict[int, int]
    # FQNs per readiness key, for diagnostics only; the key is the index of the
    # parameter in this module's trainable-parameter order.
    _grad_readiness_fqns: dict[int, tuple[str, ...]]
    # The hook registered through ``register_post_backward_hook``; the schedule
    # invokes it when ``_grad_readiness`` is in use.
    _scheduled_post_backward_hook: Callable[["FsdpModule"], None] | None
    # An under-fire at the close edge means an over-declaration or a contribution
    # that never arrived; an over-fire means the multiplicity was under-declared and
    # the window already closed early. Both are reported once per unit so a real
    # schedule bug stays visible without flooding.
    _warned_missing_grads: bool
    _warned_over_fire: bool
    _is_root: bool
    _schedule_policy: SchedulePolicy
    # Event recorded after this FsdpModule's full parameters are materialized.
    # ``None`` lets pre_forward enqueue an all-gather unless an earlier FsdpModule
    # already prefetched this module.
    _unshard_event: torch.cuda.Event | None
    # The payload orientation currently materialized for this module, or ``None``
    # when nothing is materialized. Normally a forward materializes ``"rowwise"``
    # and a backward ``"colwise"``, but activation recomputation runs a forward
    # between pre_backward() and post_backward() with no reshard in between, so
    # the backward widens the resident ``"rowwise"`` materialization to ``"both"``
    # instead of gathering into a module that is already unsharded.
    _materialized_orientation: PayloadOrientation | None
    # ``phase`` is FORWARD between pre_forward() and post_forward(), BACKWARD
    # between pre_backward() and post_backward(), and RESTING otherwise. The only
    # exception is non-reentrant activation recomputation: it runs between pre_backward()
    # and post_backward(), preserving BACKWARD through its nested forward hooks.
    _phase: Phase

    def __init__(
        self,
        context: FsdpContext,
        mesh: DeviceMesh,
        model_weight_placements: tuple[Placement, ...],
        main_grad_placements: tuple[Placement, ...],
        main_weight_placements: tuple[Placement, ...],
        mixed_precision_policy: MixedPrecisionPolicy,
        grad_divisor: int = 1,
        schedule_policy: SchedulePolicy = SchedulePolicy(),
        use_symmetric_memory: bool = False,
        register_hooks: bool = True,
        subgroup_size: int | None = None,
    ) -> None:
        """Initialize FSDP runtime state on an already-constructed module."""
        self._context = context
        self._is_root = False
        self._name = None
        self._unshard_event = None
        self._materialized_orientation = None
        self._phase = FsdpModule.Phase.RESTING
        self._schedule_policy = schedule_policy
        self._grad_readiness = None
        self._grad_readiness_fqns = {}
        self._grad_multiplicity = {}
        self._scheduled_post_backward_hook = None
        self._warned_missing_grads = False
        self._warned_over_fire = False
        owned_parameters = _collect_owned_parameters(self)
        if grad_divisor <= 0:
            raise ValueError(f"grad_divisor must be positive, got {grad_divisor}.")

        for name, parameter in owned_parameters.items():
            if is_float8tensor(parameter) and not fp8_need_transpose_data(parameter):
                raise ValueError(
                    f"MFSDP v2 only supports MXFP8 primary weights; parameter {name!r} is "
                    f"a {type(parameter).__name__} without transpose data."
                )

        parameter_groups = []
        for group_parameters in _group_parameters(owned_parameters):
            group_dtype = next(iter(group_parameters.values())).dtype
            parameter_group_cls = (
                Fp8ParameterGroup
                if all(_is_fp8_parameter(parameter) for parameter in group_parameters.values())
                else FsdpParameterGroup
            )
            parameter_groups.append(
                parameter_group_cls(
                    owning_module=self,
                    fqn_to_parameter=group_parameters,
                    mesh=mesh,
                    model_weight_placements=_specialize_placements(
                        model_weight_placements, group_dtype
                    ),
                    main_grad_placements=_specialize_placements(main_grad_placements, group_dtype),
                    main_weight_placements=_specialize_placements(
                        main_weight_placements, group_dtype
                    ),
                    mixed_precision_policy=mixed_precision_policy,
                    grad_divisor=grad_divisor,
                    use_symmetric_memory=use_symmetric_memory,
                    subgroup_size=subgroup_size,
                )
            )
        self._parameter_groups = tuple(parameter_groups)
        # The state-dict safety hook is registered unconditionally. It is still
        # required to keep loading a state dict safe when ``register_hooks`` is
        # False (i.e. execution hooks are disabled), so it must not be turned off
        # together with the auto-execution hooks below.
        self.register_load_state_dict_pre_hook(FsdpModule._pre_load_state_dict)
        if register_hooks:
            self._register_hooks()
        context.register_module(self)

    @property
    def context(self) -> FsdpContext:
        """Return the FSDP context."""
        return self._context

    @property
    def phase(self) -> Phase:
        """Return this module's lifecycle phase."""
        return self._phase

    @phase.setter
    def phase(self, phase: Phase) -> None:
        """Transition this module between its valid lifecycle phases."""
        allowed_transitions = {
            (FsdpModule.Phase.RESTING, FsdpModule.Phase.FORWARD),
            (FsdpModule.Phase.FORWARD, FsdpModule.Phase.RESTING),
            (FsdpModule.Phase.RESTING, FsdpModule.Phase.BACKWARD),
            (FsdpModule.Phase.BACKWARD, FsdpModule.Phase.RESTING),
        }
        if (self._phase, phase) not in allowed_transitions:
            raise RuntimeError(f"Invalid FSDP module phase transition: {self._phase} -> {phase}.")
        self._phase = phase

    @property
    def name(self) -> str:
        """Return this FsdpModule's name."""
        name = self._name
        if name is None:
            raise RuntimeError("FSDP module name has not been initialized.")
        return name

    def is_root(self) -> bool:
        """Return whether this module is an outermost FsdpModule in its context."""
        return self._is_root

    def _register_hooks(self) -> None:
        module = cast(nn.Module, self)
        # Use PyTorch's callback module argument instead of capturing self so
        # these hooks do not retain a deleted FSDP module.
        module.register_forward_pre_hook(
            lambda hooked_module, _args: cast(FsdpModule, hooked_module).pre_forward()
        )
        module.register_forward_hook(
            lambda hooked_module, _args, _output: cast(FsdpModule, hooked_module).post_forward()
        )
        module.register_full_backward_pre_hook(
            lambda hooked_module, _grad_output: cast(FsdpModule, hooked_module).pre_backward()
        )
        self.register_post_backward_hook(FsdpModule.post_backward)

    def register_post_backward_hook(
        self, post_backward_hook: Callable[["FsdpModule"], None], *, grad_multiplicity: bool = False
    ) -> None:
        """Register a post-backward hook to run after this module's backward completes.

        The hook runs when this module's backward is complete, so it can reshard
        this module's parameters and reduce their gradients. It is invoked once
        all of this module's trainable parameters have accumulated gradients, or
        via a full-backward hook when the module owns no trainable parameters.

        Args:
            post_backward_hook: Callback receiving this FSDP module after all of its
                trainable parameters have accumulated gradients.
            grad_multiplicity: Whether the caller is a fine-grained schedule that
                delimits backward windows itself. In that path each module's
                backward is one ``run_backward`` per schedule node, so a parameter
                contributes once per consuming node and the callback count is not
                the module's parameter count. The hook is then held until
                :meth:`finalize_scheduled_backward` is called at the schedule's
                end-of-backward edge, and the per-parameter callbacks only record
                how many contributions each parameter produced. The expected count
                of each parameter is declared separately through
                :meth:`set_grad_multiplicity`; anything undeclared is expected once.
        """
        module = cast(nn.Module, self)
        if self._num_trainable_parameters() == 0:
            module.register_full_backward_hook(
                lambda hooked_module, _grad_input, _grad_output: post_backward_hook(
                    cast(FsdpModule, hooked_module)
                )
            )
            return

        # Gradient reduction for trainable parameters is parameter-completion
        # based: once every owned Parameter has reached its declared multiplicity,
        # this FsdpModule can reduce and reshard. Module full-backward hooks can
        # fire before that when module inputs do not require grad.
        self._register_completion_hooks(post_backward_hook, scheduled=grad_multiplicity)

    def set_grad_multiplicity(self, multiplicity: Mapping[int, int]) -> None:
        """Declare how many contributions each trainable parameter is expected to make.

        Keys are indices into this module's :meth:`_trainable_fsdp_parameters`
        order, which is the same key space the per-parameter callbacks mark, and the
        values are the number of autograd GraphTasks that will accumulate that
        parameter's gradient in one backward window. Combined/fine-grained 1F1B
        supplies the real counts because its backward is one GraphTask per schedule
        node; the automatic single-graph path needs no declaration at all and
        defaults every parameter to 1.

        The declaration must be set before the hooks are registered. Declaring too
        few is the original defect (the window closes early and the surplus leaks
        into the next window), so an undeclared parameter defaults to 1 only because
        1 is the single-graph truth for the automatic path.
        """
        unknown = sorted(set(multiplicity) - set(range(self._num_trainable_parameters())))
        if unknown:
            raise ValueError(
                f"Gradient multiplicity declared for unknown parameter indices {unknown} of "
                f"{self._name or '<root>'} with {self._num_trainable_parameters()} "
                f"trainable parameters."
            )
        self._grad_multiplicity = dict(multiplicity)

    def _num_trainable_parameters(self) -> int:
        """Return the size of this module's trainable-parameter key space."""
        return sum(1 for _ in self._trainable_fsdp_parameters())

    def _register_completion_hooks(
        self, post_backward_hook: Callable[["FsdpModule"], None], *, scheduled: bool
    ) -> None:
        """Install one per-parameter accounting hook set for both completion modes.

        The hook set is identical in both modes; only the declaration differs. The
        automatic path declares 1 for every parameter, so the first callback of a
        window finds the accounting complete and triggers exactly as the fire-count
        signal used to. The schedule-driven path declares the real multiplicities
        and holds the hook until the schedule's end-of-backward edge, so a parameter
        consumed by several schedule nodes cannot end the window early.

        See :class:`MultiplicityReadiness` for why a callback count is not a valid
        completion signal in the combined/fine-grained 1F1B path.
        """
        fsdp_parameters = tuple(self._trainable_fsdp_parameters())
        declared = (getattr(self, "_grad_multiplicity", None) or {}) if scheduled else {}
        expected = {index: declared.get(index, 1) for index in range(len(fsdp_parameters))}
        readiness = MultiplicityReadiness(cast(Mapping[Hashable, int], expected))
        self._grad_readiness = readiness
        self._grad_readiness_fqns = {
            index: fsdp_parameter.fqns for index, fsdp_parameter in enumerate(fsdp_parameters)
        }
        self._scheduled_post_backward_hook = post_backward_hook
        close_on_completion = not scheduled
        module_ref = ref(self)

        def grad_hook(index: int) -> Callable[[nn.Parameter], None]:
            def hook(_: nn.Parameter) -> None:
                module = module_ref()
                if module is None:
                    return
                module._record_grad_contribution(index)
                if close_on_completion and readiness.is_complete():
                    module._close_grad_window()
                    post_backward_hook(module)

            return hook

        for index, fsdp_parameter in enumerate(fsdp_parameters):
            self._register_grad_hook(fsdp_parameter, grad_hook(index))

    def _record_grad_contribution(self, index: int) -> None:
        """Charge one gradient contribution to ``index`` in the open window.

        This is the single entry point for both the autograd callback and TE's
        delayed-wgrad callback, so the delayed path is accounted for by the same
        clock as the regular one.
        """
        assert self._grad_readiness is not None
        self._grad_readiness.mark(index)

    def _close_grad_window(self) -> None:
        """Report the window's verdict, re-arm the accounting, and mark it closed.

        Reporting happens before the reset because both the under-fire and the
        over-fire records describe the window that is ending.
        """
        readiness = self._grad_readiness
        assert readiness is not None
        if not readiness.is_complete() or readiness.over_fired():
            self._report_grad_window(readiness)
        readiness.close()

    def _report_grad_window(self, readiness: MultiplicityReadiness) -> None:
        """Report an over-fire or an under-fire of the window that is ending.

        Both directions mean the declared multiplicity did not describe the real
        GraphTask partition. Each is reported once per unit: the point is to make a
        wrong declaration loud without emitting a line per callback.

        The counts are read here, before :meth:`MultiplicityReadiness.close` resets
        them, so the message names the parameter, what was observed and what was
        declared.
        """
        over_fired = readiness.over_fired()
        if over_fired and not getattr(self, "_warned_over_fire", False):
            self._warned_over_fire = True
            self._log_grad_window_report(
                "MFSDP module %s observed MORE gradient contributions than its declared "
                "multiplicity for %d parameter(s): %s. The multiplicity was under-declared, "
                "so its backward window closed before the surplus contribution arrived; the "
                "surplus is charged to a later window. Declare the real per-parameter counts.",
                self.name or "<root>",
                len(over_fired),
                [
                    (self._grad_readiness_fqns[key], observed, expected)
                    for key, observed, expected in over_fired
                ],
            )
        missing = readiness.missing()
        if missing and not getattr(self, "_warned_missing_grads", False):
            self._warned_missing_grads = True
            self._log_grad_window_report(
                "MFSDP module %s closed its backward with %d of %d parameters under their "
                "declared multiplicity and zero-filled them: %r. Expected for a parameter "
                "unused in this window; an over-declaration or a schedule node that never ran "
                "otherwise.",
                self.name or "<root>",
                len(missing),
                len(readiness.expected),
                self._describe_grad_shortfall(readiness),
            )

    @staticmethod
    def _log_grad_window_report(message: str, *args: object) -> None:
        """Emit one window report on the reporting rank.

        A single choke point keeps the report construction testable without a
        process group (``log_single_rank`` emits on rank 0 only, while a multi-rank CI
        run may execute the test on any rank) and still logs through the module's
        normal logger in production.
        """
        log_single_rank(logger, logging.WARNING, message, *args)

    def finalize_scheduled_backward(self) -> None:
        """Close this module's backward window and reduce its gradients.

        The fine-grained schedule calls this at the point where this module's
        backward ends: from the unit's last backward node, or from the end of the
        model chunk for the module that owns the chunk-level parameters. The
        per-parameter marks collected since the previous close are counted against
        the declared multiplicity, so a parameter consumed by several schedule nodes
        cannot end the window early and a parameter that never contributed is
        observable instead of silently satisfied.

        A parameter that came up short is genuinely unused in this window, so its
        contribution is an exact zero and its slot is zero-filled rather than
        aborting the module's reduce. The reduce is still issued by every rank, so
        the decision is collective-safe. A module whose window never opened is left
        alone, exactly as before.
        """
        if not self._has_open_grad_window():
            return
        self._close_grad_window()
        assert self._scheduled_post_backward_hook is not None
        self._scheduled_post_backward_hook(self)

    def assert_scheduled_backward_closed(self) -> None:
        """Raise if the schedule ended while this module still had a gradient pending.

        An open window at the end of a chunk means the module received gradients
        that no end-of-backward edge ever consumed, which is the "reduce-scatter
        silently dropped" failure mode of a desynchronised completion signal. An
        over-fire is reported the same way: it means the window already closed
        early, so a later contribution re-opened it.
        """
        readiness = self._grad_readiness
        if not self._has_open_grad_window():
            return
        assert readiness is not None
        missing = self._describe_grad_shortfall(readiness)
        over_fired = [
            (self._grad_readiness_fqns[key], observed, expected)
            for key, observed, expected in readiness.over_fired()
        ]
        raise RuntimeError(
            "MFSDP schedule ended with an incomplete gradient-completion window for "
            f"{self.name or '<root>'}: marks={readiness.marked_total}/"
            f"{readiness.expected_total}, missing={missing!r}, over_fired={over_fired!r}."
        )

    def _describe_grad_shortfall(self, readiness: MultiplicityReadiness) -> list[tuple]:
        """Return ``(fqns, observed, expected)`` for every under-fired parameter."""
        return [
            (self._grad_readiness_fqns[key], readiness.count(key), readiness.expected[key])
            for key in sorted(readiness.missing())
        ]

    def _has_open_grad_window(self) -> bool:
        """Return whether this module's accounting holds unconsumed contributions.

        A window is open exactly when at least one contribution has arrived since
        the last close. A window in which nothing fired is a no-op, matching the
        previous behaviour: nothing was reduced, so there is nothing to reset.
        """
        readiness = self._grad_readiness
        if readiness is None:
            return False
        return readiness.has_pending_marks

    def _trainable_fsdp_parameters(self) -> Iterator[FsdpParameter]:
        """Iterate this module's trainable parameters in a stable order."""
        for group in self._parameter_groups:
            if not group.requires_grad:
                continue
            yield from group.fsdp_parameters

    def _register_grad_hook(
        self, fsdp_parameter: FsdpParameter, hook: Callable[[nn.Parameter], None]
    ) -> None:
        """Register ``hook`` on the representation that actually produces the gradient."""
        parameter = fsdp_parameter.unsharded
        # ``skip_backward_post_hook`` is TE's delayed-wgrad contract: these
        # gradients are materialized by ``backward_dw()``, not autograd.
        if not getattr(parameter, "skip_backward_post_hook", False):
            parameter.register_post_accumulate_grad_hook(hook)
            return
        if len(fsdp_parameter.fqns) > 1:
            raise ValueError(
                "Tied parameters with delayed wgrad are not supported because "
                "Transformer Engine does not accumulate their gradients. See "
                "https://github.com/NVIDIA/TransformerEngine/issues/3437"
            )
        parameter_module, _ = get_parameter_owner(cast(nn.Module, self), fsdp_parameter.fqns[0])
        parameter_module.register_wgrad_accumulation_and_reduce_hooks(
            lambda parameter=parameter: hook(parameter)
        )

    @staticmethod
    def _pre_load_state_dict(
        _module: nn.Module,
        _state_dict: dict[str, object],
        _prefix: str,
        local_metadata: dict[str, object],
        _strict: bool,
        _missing_keys: list[str],
        _unexpected_keys: list[str],
        _error_msgs: list[str],
    ) -> None:
        """Reject state-dict loads that replace parameters managed by FSDP."""
        # PyTorch propagates load_state_dict(assign=True) to pre-hooks through
        # this internal metadata key before replacing parameters and buffers.
        if local_metadata.get("assign_to_params_buffers", False):
            raise RuntimeError(
                "load_state_dict(assign=True) is not supported after fully_shard(). "
                "Load before fully_shard() or use an in-place load path with assign=False."
            )

    def pre_forward(self) -> None:
        """Prepare full parameters for forward compute and prefetch the next FsdpModule.

        While this FsdpModule computes, we issue the next FsdpModule's all-gather
        on the comm stream, so ``AG_{i+1}`` is launched before ``F_i`` finishes.
        """
        context = self.context
        # This is the first MFSDP hook to run, so finalize the context here once
        # before any module begins communication.
        context.ensure_finalized()
        # A reentrant checkpoint recomputes before the child module's backward-pre
        # hook runs. The active autograd GraphTask identifies that recomputation.
        is_recomputing = self.phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
        if self.phase is not FsdpModule.Phase.BACKWARD:
            self.phase = FsdpModule.Phase.FORWARD
        # forward/backward each span multiple lifecycle methods (pre_forward ->
        # post_forward and pre_backward -> post_backward), so they keep explicit
        # push/pop instead of a single context scope.
        torch.cuda.nvtx.range_push(self._nvtx_label("forward"))

        if self.is_root():
            context.allgather_stream.wait_stream(context.current_stream())

        self.unshard(prefetch="forward" if not is_recomputing else "none", orientation=ROWWISE)

    def unshard(
        self,
        prefetch: Literal["forward", "backward", "none"] = "none",
        orientation: PayloadOrientation = BOTH,
    ) -> None:
        """Unshard this FsdpModule's parameter groups immediately.

        External schedulers invoking this directly (rather than through the
        automatic ``pre_forward`` hook) must first synchronize the all-gather
        stream on the root by calling
        ``context.allgather_stream.wait_stream(context.current_stream())``
        before this when ``self.is_root()``; the automatic forward path
        performs that root sync in ``pre_forward()`` immediately before this.

        Args:
            prefetch: Static order to prefetch successors from, if any.
            orientation: Payload orientation to materialize for MXFP8 primary
                weights -- ``"rowwise"`` on a forward pass, ``"colwise"`` on a
                backward pass, ``"both"`` when one materialization has to serve
                both. Ignored by regular parameter groups. The automatic
                forward/backward paths narrow this; the ``"both"`` default is the
                safe superset for a caller that does not know the pass, and a
                request narrower than what is already materialized is a no-op.
        """
        with self._nvtx_range("unshard"):
            self._unshard_parameter_groups(orientation)
            assert self._unshard_event is not None
            # Compute waits only for this FsdpModule's all-gather (the prefetch below is
            # issued afterwards, so it is free to run concurrently with this FsdpModule).
            self.context.current_stream().wait_event(self._unshard_event)

            context = self.context
            if prefetch == "forward":
                self._prefetch_parameter_groups(
                    context.forward_order, self._schedule_policy.forward_prefetch_size, orientation
                )
            elif prefetch == "backward":
                self._prefetch_parameter_groups(
                    context.backward_order,
                    self._schedule_policy.backward_prefetch_size,
                    orientation,
                )

    def _prefetch_parameter_groups(
        self,
        order: IndexedOrder["FsdpModule"],
        prefetch_size: int | None,
        orientation: PayloadOrientation = BOTH,
    ) -> None:
        """Prefetch successors from ``order`` according to this module's budget.

        ``orientation`` is the payload those successors will be consumed with: a
        module prefetched from the forward order is about to run its forward
        (row-wise), and one prefetched from the backward order its backward
        (column-wise). Using the consumer's orientation keeps the later demand
        unshard a no-op instead of a second, redundant all-gather.
        """
        next_module = order.next_item(self)
        if prefetch_size is None:
            if next_module is not None:
                next_module._unshard_parameter_groups(orientation)
            return

        prefetched_size = 0
        while next_module is not None and prefetched_size < prefetch_size:
            next_module._unshard_parameter_groups(orientation)
            prefetched_size += next_module.num_parameter_elements
            next_module = order.next_item(next_module)

    def _unshard_parameter_groups(self, orientation: PayloadOrientation = BOTH) -> None:
        """Unshard this FsdpModule's parameter groups on the all-gather stream.

        If a compatible materialization is already resident, this method is a
        no-op: a request no wider than what was materialized (a demand unshard
        after a same-orientation prefetch, or a backward after a forward that was
        not resharded) is served as is. A request for a direction that is *not*
        resident widens the materialization, because a module's forward and
        backward unshards can share one residency window (activation
        recomputation runs a forward between ``pre_backward`` and
        ``post_backward``). Otherwise this method records ``_unshard_event`` after
        materialization so compute can wait without depending on later release
        work.

        Args:
            orientation: Payload orientation to gather for MXFP8 groups --
                ``"rowwise"`` on a forward pass, ``"colwise"`` on a backward
                pass, ``"both"`` when one materialization has to serve both.
                Ignored by regular groups.
        """
        requested = orientation_directions(orientation)
        if self._unshard_event is not None:
            resident = orientation_directions(self._materialized_orientation or BOTH)
            if requested <= resident:
                return
            # Widen in place: the groups gather only the directions still missing.
            orientation = merge_orientations(resident | requested)

        allgather_stream = self.context.allgather_stream
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.unshard_parameters(orientation)
            self._unshard_event = allgather_stream.record_event()
            self._materialized_orientation = merge_orientations(orientation_directions(orientation))

    def post_forward(self) -> None:
        """Return parameters to their sharded resting state after forward compute."""
        # Recomputed parameters are consumed immediately by this module's
        # backward. Keep them materialized to avoid an unnecessary all-gather;
        # post_backward() will reshard them after gradient reduction.
        is_recomputing = self.phase is FsdpModule.Phase.BACKWARD or _is_in_backward()
        if not is_recomputing:
            self.reshard()
        if self.phase is FsdpModule.Phase.FORWARD:
            self.phase = FsdpModule.Phase.RESTING
        torch.cuda.nvtx.range_pop()

    def reshard(self) -> None:
        """Reshard this FsdpModule's parameter groups."""
        with self._nvtx_range("reshard"):
            self._reshard_parameter_groups()

    def _reshard_parameter_groups(self) -> None:
        """Reshard parameter groups and release unsharded storage after compute.

        This method clears ``_unshard_event`` after queuing the release, so
        future users enqueue a fresh all-gather.
        """
        for group in self._parameter_groups:
            group.reshard_parameters()

        allgather_stream = self.context.allgather_stream
        allgather_stream.wait_stream(self.context.current_stream())
        # Release on the all-gather stream where unsharded storage was allocated,
        # so no record_stream() call is required for the storage.
        with torch.cuda.stream(allgather_stream):
            for group in self._parameter_groups:
                group.release_unsharded_storage()
            self._unshard_event = None
            self._materialized_orientation = None

    def pre_backward(self) -> None:
        """Prepare full parameters and prefetch the next FsdpModule in backward order."""
        self.phase = FsdpModule.Phase.BACKWARD
        torch.cuda.nvtx.range_push(self._nvtx_label("backward"))
        context = self.context
        current_stream = context.current_stream()
        if self.is_root():
            context.register_post_backward_hook()
            # Fork the reduce-scatter stream from the current stream once, at the
            # start of backward, so every module's post-backward reduce-scatter is
            # part of any active CUDA-graph capture. A stream only joins the
            # capture via this wait_stream edge; without it the first allocation on
            # the reduce-scatter stream falls back to a raw cudaMalloc, which is
            # illegal during capture. Later modules are covered by the post-copy
            # fork each preceding module issues before its collective.
            context.reduce_scatter_stream.wait_stream(current_stream)

        self.unshard(prefetch="backward", orientation=COLWISE)

    def post_backward(self) -> None:
        """Reduce gradients and return parameters to their sharded resting state."""
        self.reshard()
        self._reduce_gradient_groups()
        self.phase = FsdpModule.Phase.RESTING
        torch.cuda.nvtx.range_pop()

    def _reduce_gradient_groups(self) -> None:
        """Pack gradients and immediately launch their reduce-scatters."""
        with self._nvtx_range("reduce_gradients"):
            context = self.context
            reduce_scatter_stream = context.reduce_scatter_stream
            current_stream = context.current_stream()

            for group in self._parameter_groups:
                if not group.requires_grad:
                    continue

                # A schedule-driven module was closed by the schedule itself, so a
                # parameter without a gradient here is one that was unused in the
                # window and gets a zero-filled slot. The automatic path keeps the
                # strict check, where a missing gradient means the reduce is early.
                with torch.cuda.stream(reduce_scatter_stream):
                    partial_grad = group.allocate_partial_grad_buffer(
                        require_all_grads=self._grad_readiness is None
                    )

                current_stream.wait_stream(reduce_scatter_stream)
                group.copy_gradients_to_partial_buffer(partial_grad)

                reduce_scatter_stream.wait_stream(current_stream)
                with torch.cuda.stream(reduce_scatter_stream):
                    group.reduce_partial_gradients(
                        partial_grad, is_last_microbatch=self.context.is_last_microbatch
                    )

    @property
    def parameter_groups(self) -> tuple[FsdpParameterGroup, ...]:
        """Parameter groups owned by this FsdpModule."""
        return self._parameter_groups

    @property
    def num_parameter_elements(self) -> int:
        """Return the number of unsharded parameter elements owned by this module."""
        return sum(
            parameter.unsharded.numel()
            for group in self._parameter_groups
            for parameter in group.fsdp_parameters
        )

    def _nvtx_label(self, operation: str) -> str:
        module_name = self.name or "<root>"
        return f"MFSDP {module_name} {operation}"

    @contextmanager
    def _nvtx_range(self, operation: str) -> Iterator[None]:
        """Scope an nvtx range to this context so an early return still pops it."""
        torch.cuda.nvtx.range_push(self._nvtx_label(operation))
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()


def _collect_backward_order(module: nn.Module, order: IndexedOrder["FsdpModule"]) -> None:
    """Collect one root's static backward prefetch order."""
    if isinstance(module, FsdpModule):
        order.append(module)

    for child in reversed(list(module.children())):
        _collect_backward_order(child, order)


def _collect_fsdp_children(module: nn.Module, children: set["FsdpModule"]) -> None:
    """Collect the nearest FSDP descendants of ``module``."""
    for child in module.children():
        if isinstance(child, FsdpModule):
            children.add(child)
        else:
            _collect_fsdp_children(child, children)


def _collect_owned_parameters(root_module: nn.Module) -> dict[str, nn.Parameter]:
    parameters: dict[str, nn.Parameter] = {}

    def visit(submodule: nn.Module, submodule_fqn: str) -> None:
        direct_parameters = submodule.named_parameters(recurse=False, remove_duplicate=False)

        for local_parameter_name, parameter in direct_parameters:
            parameter_fqn = (
                f"{submodule_fqn}.{local_parameter_name}" if submodule_fqn else local_parameter_name
            )
            if get_containing_parameter_group(parameter) is not None:
                raise ValueError(
                    f"Parameter {parameter_fqn!r} is already owned by another FsdpModule."
                )
            parameters[parameter_fqn] = parameter

        for child_name, child_module in submodule.named_children():
            if isinstance(child_module, FsdpModule):
                continue
            child_fqn = f"{submodule_fqn}.{child_name}" if submodule_fqn else child_name
            visit(child_module, child_fqn)

    visit(root_module, "")
    return parameters


def _group_parameters(parameters: dict[str, nn.Parameter]) -> list[dict[str, nn.Parameter]]:
    grouped: dict[tuple[torch.dtype, bool, bool], dict[str, nn.Parameter]] = {}
    for name, parameter in parameters.items():
        key = (parameter.dtype, parameter.requires_grad, _is_fp8_parameter(parameter))
        grouped.setdefault(key, {})[name] = parameter
    return [grouped[key] for key in grouped]


def _specialize_placements(
    placements: tuple[Placement, ...], dtype: torch.dtype
) -> tuple[Placement, ...]:
    """Specialize public placements for one homogeneous parameter group.

    Today every parameter group maps Torch's user-facing ``Shard(0)`` to the
    DBuffer-specific ``Flat`` format. This dtype-homogeneous group boundary is
    where MXFP8 groups will instead select ``BlockAtomic``.
    """
    if dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise NotImplementedError(f"Unsupported dtype: {dtype}.")
    for placement in placements:
        if type(placement) is Shard and placement.dim != 0:
            raise NotImplementedError(
                "MFSDP currently supports only dim-0 Shard placements, " f"got {placement!r}."
            )
    return tuple(Flat() if type(placement) is Shard else placement for placement in placements)
