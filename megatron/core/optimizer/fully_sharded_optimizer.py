# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

import logging
import os
from typing import Callable, List, NamedTuple, Optional, override

import torch
from torch.distributed.tensor import DTensor

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    FsdpParameterGroup,
    get_containing_parameter_group,
    sync_model_weights_from_main_weights,
)
from ..transformer.module import MegatronModule
from ..utils import is_te_min_version
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


def count_replication(tensor: DTensor) -> int:
    """Return how many ranks hold an identical copy of ``tensor``'s local shard.

    A sharded mesh axis holds disjoint pieces that must all be counted; a replicated
    axis holds identical copies that must be counted once, so a gradient statistic
    summed over the grad-stats group has to divide by this.

    MFSDP v2 gradients are always DTensors, so this takes one rather than accepting
    a plain tensor and guessing a layout for it.
    """
    replication = 1
    for axis, placement in enumerate(tensor.placements):
        if placement.is_replicate():
            replication *= tensor.device_mesh.size(axis)
        elif placement.is_partial():
            raise RuntimeError(
                "MFSDP v2 gradient is still Partial when gradient statistics are taken; "
                "the reduction must be finalized first."
            )
    return replication


# Diagnostic gate for the pre-step gradient-placement check. It defaults off so a production
# run behaves exactly as before; set MFSDP_CHECK_GRAD_PLACEMENTS=1 to arm it for a run.
MFSDP_CHECK_GRAD_PLACEMENTS_ENV = "MFSDP_CHECK_GRAD_PLACEMENTS"
_TRUTHY_ENV_VALUES = frozenset(("1", "true", "yes", "on"))

# Staleness flags reported on a violation: the bf16 model-weight/main-grad flags every group
# has, plus the fp8 payload flags Fp8ParameterGroup adds.
_STALENESS_FLAG_NAMES = (
    "_model_weight_is_stale",
    "_main_grad_is_stale",
    "_rowwise_is_stale",
    "_colwise_is_stale",
)

# Bound the violation report so one failing step still prints a readable message.
_MAX_REPORTED_VIOLATIONS = 16


class GradPlacementCheckResult(NamedTuple):
    """What one pre-step gradient-placement check observed."""

    checked: int
    skipped_no_grad: int
    observed: tuple[tuple[str, str], ...]


def grad_placement_check_enabled() -> bool:
    """Whether the pre-step MFSDP v2 gradient-placement diagnostic is armed."""
    value = os.environ.get(MFSDP_CHECK_GRAD_PLACEMENTS_ENV, "")
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _format_placement(placement: object) -> str:
    """Render one placement, keeping ``Flat``/``BlockAtomic`` distinguishable."""
    details = []
    dim = getattr(placement, "dim", None)
    if dim is not None:
        details.append(f"dim={dim}")
    block_size = getattr(placement, "block_size", None)
    if block_size is not None:
        details.append(f"block_size={block_size}")
    suffix = f"({', '.join(details)})" if details else "()"
    return f"{type(placement).__name__}{suffix}"


def _format_placements(placements: object) -> str:
    """Render a placements sequence, or a placeholder when it is absent."""
    if placements is None:
        return "<none>"
    try:
        return "[" + ", ".join(_format_placement(placement) for placement in placements) + "]"
    except TypeError:
        return f"<unavailable:{type(placements).__name__}>"


def _format_buffer_placements(parameter_group: object, name: str) -> str:
    """Render one named optimizer-layout buffer's placements on a parameter group."""
    buffer = getattr(parameter_group, name, None)
    if buffer is None:
        return f"{name}.placements=<buffer-none>"
    return f"{name}.placements={_format_placements(getattr(buffer, 'placements', None))}"


def _describe_parameter_group(parameter_group: FsdpParameterGroup | None) -> str:
    """Collect the parameter-group state that explains a placement mismatch."""
    if parameter_group is None:
        return "parameter_group=<none>"
    details = [
        _format_buffer_placements(parameter_group, name)
        for name in ("main_weight", "main_grad", "pre_optimizer_main_grad")
    ]
    mesh = getattr(parameter_group, "mesh", None)
    mesh_tensor = getattr(mesh, "mesh", None)
    mesh_size = getattr(mesh, "size", None)
    details.append(
        f"mesh.shape={tuple(mesh_tensor.shape) if mesh_tensor is not None else None} "
        f"mesh.size={mesh_size() if callable(mesh_size) else None}"
    )
    details.extend(
        f"{name}={getattr(parameter_group, name, '<absent>')}" for name in _STALENESS_FLAG_NAMES
    )
    return " ".join(details)


def _parameter_fqns(parameter_group: FsdpParameterGroup | None, parameter: object) -> str:
    """Return the FQNs the owning group registered for ``parameter``."""
    if parameter_group is None:
        return "<unknown:no-parameter-group>"
    for fsdp_parameter in parameter_group.fsdp_parameters:
        if fsdp_parameter.sharded is parameter:
            return str(fsdp_parameter.fqns)
    return "<unknown:not-in-parameter-group>"


def check_gradient_placements(
    parameters: List[torch.nn.Parameter], *, optimizer_label: str
) -> GradPlacementCheckResult:
    """Verify every stepped gradient is in its owning group's optimizer layout.

    Dense HFSDP masters are ``[Shard, Shard]``, expert ZeRO-1 is a single-axis ``[Shard]``
    and single-axis ZeRO-3 is ``[Shard]``, so the expected value is read per group from
    ``main_weight`` rather than hardcoded. ``ChainedOptimizer.step()`` first reaches
    ``count_replication(parameter.grad)``, which rejects a ``Partial`` gradient but accepts
    every other layout, so a gradient left in the *parameter* layout while the masters moved
    to the optimizer layout would step a tensor the peers do not agree on. That is the class
    of bug this check exists to catch before the optimizer touches the gradient.

    Args:
        parameters: Parameters the optimizer is about to step.
        optimizer_label: Human-readable identity of the optimizer being checked, included in
            both the failure message and the once-per-rank summary.

    Returns:
        A :class:`GradPlacementCheckResult` with the counts and the distinct
        ``(grad, main_weight)`` placement pairs observed.

    Raises:
        RuntimeError: If a parameter that is about to be stepped has a gradient outside its
            owning group's optimizer layout, or belongs to no parameter group at all.
    """
    checked = 0
    skipped_no_grad = 0
    observed: set[tuple[str, str]] = set()
    violations: List[str] = []
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            # A parameter with no gradient is not stepped, so its layout cannot skew a
            # collective. Count it so the summary still shows how much was skipped.
            skipped_no_grad += 1
            continue
        checked += 1
        parameter_group = get_containing_parameter_group(parameter)
        grad_placements = getattr(grad, "placements", None)
        expected_placements = (
            None
            if parameter_group is None
            else getattr(getattr(parameter_group, "main_weight", None), "placements", None)
        )
        observed.add((_format_placements(grad_placements), _format_placements(expected_placements)))
        if parameter_group is None:
            violations.append(
                f"fqns={_parameter_fqns(parameter_group, parameter)} parameter_group=<none> "
                f"grad.type={type(grad).__name__} "
                f"grad.placements={_format_placements(grad_placements)}"
            )
            continue
        if grad_placements is None or tuple(grad_placements) != tuple(expected_placements):
            violations.append(
                f"fqns={_parameter_fqns(parameter_group, parameter)} "
                f"grad.type={type(grad).__name__} "
                f"grad.placements={_format_placements(grad_placements)} "
                f"expected(main_weight).placements={_format_placements(expected_placements)} "
                f"{_describe_parameter_group(parameter_group)}"
            )

    if violations:
        hidden = len(violations) - _MAX_REPORTED_VIOLATIONS
        shown = violations[:_MAX_REPORTED_VIOLATIONS]
        if hidden > 0:
            shown.append(f"+{hidden} more violating parameter(s) not shown")
        raise RuntimeError(
            "MFSDP v2 gradient-placement check failed for "
            f"optimizer={optimizer_label}: parameter.grad must be in the owning parameter "
            "group's optimizer layout (param.grad.placements == main_weight.placements) "
            f"before the optimizer step. checked={checked} skipped_no_grad={skipped_no_grad} "
            f"violations={len(violations)}: " + " | ".join(shown)
        )

    return GradPlacementCheckResult(
        checked=checked, skipped_no_grad=skipped_no_grad, observed=tuple(sorted(observed))
    )


class FullyShardedOptimizer(MixedPrecisionOptimizer):
    """MCore optimizer wrapper for MFSDP-owned sharded parameters and gradients.

    MFSDP v2 owns the optimizer-facing parameter and gradient shards directly.
    Unlike :class:`DistributedOptimizer`, this wrapper does not build DDP
    param-and-grad-buffer range maps or allocate separate main-parameter shards.
    It preserves MCore's mixed-precision optimizer step contract while making
    MFSDP-specific storage operations explicit.
    """

    # ChainedOptimizer's combined gradient-statistics path requires every DTensor
    # to use the same device mesh. MFSDP needs the per-optimizer implementation
    # below so dense and expert parameters on different meshes are counted
    # correctly, even when their final reduction process group is the same.
    requires_individual_grad_stats = True

    @override
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: Optional[MegatronGradScaler],
        init_state_fn: Callable,
        model_chunks: List[MegatronModule],
    ) -> None:
        """Initialize the MFSDP optimizer wrapper.

        Args:
            optimizer: Base optimizer such as Adam or SGD.
            config: Optimizer configuration.
            grad_scaler: Optional loss scaler. Currently unsupported for MFSDP v2,
                but accepted to match the MCore optimizer construction contract.
            init_state_fn: Function used to initialize optimizer state.
            model_chunks: MFSDP v2 model chunks optimized by this wrapper.
        """
        FullyShardedOptimizer._validate_config(config, model_chunks)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)
        if grad_scaler is not None:
            raise ValueError("MFSDP v2 does not currently support loss scaling.")

        super().__init__(optimizer, config, grad_scaler, init_state_fn)
        self.model_chunks = model_chunks
        self.ddp_config = self.model_chunks[0].ddp_config
        for model_chunk in self.model_chunks:
            if self.ddp_config != model_chunk.ddp_config:
                raise ValueError("All MFSDP v2 model chunks must share the same ddp_config.")
        self.is_stub_optimizer = optimizer is None
        self._casted_grads = []
        self._grad_placement_summary_logged = False

    @staticmethod
    def _validate_config(config: OptimizerConfig, model_chunks: List[MegatronModule]) -> None:
        """Validate the MFSDP v2 optimizer support contract."""
        # Multiple model chunks are allowed: VPP shares a single FsdpContext across
        # chunks, and FullyShardedOptimizer optimizes every chunk's parameters
        # together (self.model_chunks is iterated in zero_grad / get_parameters).
        if not model_chunks:
            raise ValueError("MFSDP v2 requires at least one model chunk.")
        if config.use_distributed_optimizer:
            raise ValueError("MFSDP v2 currently requires use_distributed_optimizer=False.")
        if config.loss_scale is not None:
            raise ValueError("MFSDP v2 does not currently support loss scaling.")
        if config.fp16:
            raise ValueError(
                "MFSDP v2 does not currently support FP16 training because FP16 triggers "
                "loss unscale."
            )
        if config.overlap_param_gather_with_optimizer_step:
            raise ValueError("MFSDP v2 does not support optimizer-step parameter-gather overlap.")
        if config.optimizer_cpu_offload:
            raise ValueError("MFSDP v2 does not currently support optimizer CPU offload.")
        if config.use_layer_wise_distributed_optimizer:
            raise ValueError(
                "MFSDP v2 does not currently support layer-wise distributed optimizer."
            )

    @override
    def state_dict(self):
        """Return optimizer state.

        MFSDP v2 optimizer checkpointing needs an FSDP-native DTensor state
        contract. Keep this intentionally unsupported for the prototype instead
        of falling back to DDP-buffer assumptions.
        """
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    @override
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    @override
    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Build a sharded optimizer state dict."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    @override
    def get_grad_norm(self):
        """Compute the global gradient L2 norm from each gradient's own DTensor layout.

        MFSDP v2 gradients are DTensors that record how they are distributed, and the
        dense and expert gradients do not share a device mesh: with EP=2 over eight
        ranks the dense gradients live on all eight while the expert gradients live on
        the four-rank expert-DP stripe. Reading the layout off each gradient keeps the
        norm correct without assuming a single mesh for all of them.

        Each rank contributes ``||local||^2`` divided by the product of its replicated
        mesh-axis sizes. A sharded axis holds disjoint pieces that must all be added; a
        replicated axis holds identical copies that must be counted once. Summing that
        over the grad-stats group is then exact, because every shard is held by exactly
        one rank in that group.

        ``get_grad_norm_fp32`` cannot do this: ``get_main_grads_for_grad_norm``
        replaces each DTensor with ``grad._local_tensor`` before it runs, so
        ``get_data_parallel_group_if_dtensor`` always sees plain tensors, returns None,
        and the layout is gone by the time the norm is taken.
        """
        total_norm_squared = torch.zeros(
            (), dtype=torch.float32, device=torch.cuda.current_device()
        )
        for parameter in self.get_parameters():
            # MFSDP v2 reduces into parameter.grad; it never populates decoupled_grad,
            # which is a v1 param-and-grad-buffer concept.
            grad = parameter.grad
            if grad is None:
                continue
            replication = count_replication(grad)
            local_grad = grad.to_local()
            if local_grad.numel() > 0:
                total_norm_squared += local_grad.float().pow(2).sum() / replication

        torch.distributed.all_reduce(
            total_norm_squared,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_grad_stats_parallel_group(),
        )
        return total_norm_squared.sqrt()

    @override
    def count_zeros(self) -> float:
        """Count zero gradient entries from each gradient's own DTensor layout.

        ``count_zeros_fp32`` has the same single-mesh assumption as the grad-norm path,
        and additionally rejects the combination of a Megatron-FSDP parameter with a
        DTensor-derived data-parallel group. Counting here keeps MFSDP v2 off that path,
        and matches how ``get_grad_norm`` reduces: each rank contributes its own shard,
        divided by the size of any replicated mesh axis, summed over the grad-stats group.
        """
        total_zeros = torch.zeros((), dtype=torch.float32, device=torch.cuda.current_device())
        for parameter in self.get_parameters():
            grad = parameter.grad
            if grad is None:
                continue
            replication = count_replication(grad)
            local_grad = grad.to_local()
            if local_grad.numel() > 0:
                zeros = local_grad.numel() - torch.count_nonzero(local_grad)
                total_zeros += zeros.float() / replication

        torch.distributed.all_reduce(
            total_zeros,
            op=torch.distributed.ReduceOp.SUM,
            group=self.get_grad_stats_parallel_group(),
        )
        return total_zeros.item()

    @override
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear optimizer-visible sharded grads."""
        # install_sharded_grads() binds .grad to a persistent main_grad view from
        # Python during backward, and graph replay re-executes only GPU kernels. So
        # unbinding here is never undone on a replayed step: .grad stays None, the
        # optimizer skips every parameter, and the run silently stops training with
        # a 0.0 grad norm. Zero in place to keep the binding alive.
        #
        # Overriding rather than asserting: ChainedOptimizer forwards its own True
        # default positionally, so set_to_none is always True here regardless of
        # caller intent, and an assert would fire on every graphed step.
        if any(model_chunk.config.cuda_graph_impl != "none" for model_chunk in self.model_chunks):
            set_to_none = False

        if not self.is_stub_optimizer:
            self.optimizer.zero_grad(set_to_none=set_to_none)

        if not is_te_min_version("2.18.0"):
            # Older TE FusedAdam requires empty local shards to be omitted from optimizer
            # parameter groups (see the matching workaround in get_megatron_optimizer()).
            # Those parameters are not cleared by optimizer.zero_grad().
            for model_chunk in self.model_chunks:
                model_chunk.zero_grad(set_to_none=set_to_none)

    def _copy_model_grads_to_main_grads(self) -> None:
        """Install optimizer-compatible gradients for non-precision-aware optimizers."""
        if self.config.use_precision_aware_optimizer:
            return

        assert not self._casted_grads
        for parameter in self.get_parameters():
            if parameter.grad is None:
                continue
            if parameter.grad.dtype == parameter.data.dtype:
                continue

            original_grad = parameter.grad
            parameter.grad = None
            parameter.grad_dtype = parameter.data.dtype
            parameter.grad = original_grad.to(dtype=parameter.data.dtype)
            self._casted_grads.append((parameter, original_grad))

    @override
    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer and restore MFSDP gradient dtypes."""
        self._check_gradient_placements()
        success = super().step_with_ready_grads()
        for parameter, original_grad in self._casted_grads:
            parameter.grad = None
            parameter.grad_dtype = original_grad.dtype
            parameter.grad = original_grad
        self._casted_grads.clear()
        return success

    def _check_gradient_placements(self) -> None:
        """Run the opt-in pre-step gradient-placement diagnostic for this optimizer.

        Runs once per optimizer step, immediately before any optimizer work.
        ``ChainedOptimizer._step`` calls ``step_with_ready_grads`` on every chained optimizer,
        and MFSDP v2 wraps both the Muon optimizer and the Adam/scalar optimizer for excluded
        parameters in a :class:`FullyShardedOptimizer`, so this covers both with no
        special-casing. The summary is logged at most once per optimizer per rank: never once
        per parameter and never once per step.
        """
        if not grad_placement_check_enabled():
            return

        optimizer_label = f"{type(self).__name__}(inner={type(self.optimizer).__name__})"
        result = check_gradient_placements(self.get_parameters(), optimizer_label=optimizer_label)
        if self._grad_placement_summary_logged:
            return
        self._grad_placement_summary_logged = True
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 0
        )
        logger.warning(
            "MFSDP grad-placement check passed rank=%s optimizer=%s checked=%s "
            "skipped_no_grad=%s distinct(grad,main_weight)_placements=%s",
            rank,
            optimizer_label,
            result.checked,
            result.skipped_no_grad,
            result.observed,
        )

    def _copy_main_params_to_model_params(self) -> None:
        """Refresh MFSDP V2 compute weights after updating optimizer weights."""
        sync_model_weights_from_main_weights(self.get_parameters())

    def _copy_model_params_to_main_params(self, state_dict=None) -> None:
        """No-op: model loads already write into MFSDP v2's main weights."""
