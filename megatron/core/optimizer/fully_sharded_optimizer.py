# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

import os
from typing import Callable, List, Optional

import torch

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    sync_model_weights_from_main_weights,
)
from ..transformer.module import MegatronModule
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer
from .optimizer_config import OptimizerConfig


class FullyShardedOptimizer(MixedPrecisionOptimizer):
    """MCore optimizer wrapper for MFSDP-owned sharded parameters and gradients.

    MFSDP v2 owns the optimizer-facing parameter and gradient shards directly.
    Unlike :class:`DistributedOptimizer`, this wrapper does not build DDP
    param-and-grad-buffer range maps or allocate separate main-parameter shards.
    It preserves MCore's mixed-precision optimizer step contract while making
    MFSDP-specific storage operations explicit.
    """

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
        contexts = {model_chunk.context for model_chunk in self.model_chunks}
        if len(contexts) != 1:
            raise ValueError("All MFSDP v2 model chunks must share one FsdpContext.")
        self.context = contexts.pop()
        self.is_stub_optimizer = optimizer is None
        self._casted_grads = []
        self._nan_diagnostic_step = 0

    @torch.no_grad()
    def _run_nan_diagnostic(self, phase: str) -> None:
        """Locate the first non-finite FSDP buffer or optimizer state.

        This temporary diagnostic is gated by an environment variable and is
        intentionally kept out of the production PR. It scans only the final
        few requested iterations, using foreach infinity norms so the check
        does not allocate one boolean per model element.
        """
        start_step = int(os.environ.get("MCORE_MFSDP_NAN_DIAGNOSTIC_START_STEP", "0"))
        if start_step <= 0 or self._nan_diagnostic_step < start_step:
            return

        entries = []
        for chunk_index, model_chunk in enumerate(self.model_chunks):
            for module in model_chunk.modules():
                for group_index, parameter_group in enumerate(
                    getattr(module, "parameter_groups", ())
                ):
                    module_name = getattr(module, "_name", type(module).__qualname__)
                    prefix = f"chunk={chunk_index} module={module_name!r} group={group_index}"
                    entries.append((f"{prefix} main_weight", parameter_group.main_weight.local_buffer))
                    main_grad = getattr(parameter_group, "_main_grad", None)
                    if main_grad is not None:
                        entries.append((f"{prefix} main_grad", main_grad.local_buffer))

        if not self.is_stub_optimizer:
            parameter_labels = {}
            for group_index, param_group in enumerate(self.optimizer.param_groups):
                for parameter_index, parameter in enumerate(param_group["params"]):
                    parameter_labels[parameter] = (
                        f"optimizer_group={group_index} parameter={parameter_index}"
                    )
            for parameter, state in self.optimizer.state.items():
                prefix = parameter_labels.get(parameter, "optimizer_parameter=unknown")
                for state_name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        entries.append((f"{prefix} state={state_name}", value))

        # Foreach kernels require homogeneous device/dtype lists. DTensor-backed
        # optimizer state is inspected through its local tensor.
        buckets = {}
        for label, value in entries:
            value = getattr(value, "_local_tensor", value)
            if value is None or not value.is_cuda or not value.is_floating_point():
                continue
            buckets.setdefault((value.device, value.dtype), []).append((label, value.reshape(-1)))

        bad_labels = []
        largest = []
        for bucket in buckets.values():
            labels = [label for label, _ in bucket]
            values = [value for _, value in bucket]
            norms = torch._foreach_norm(values, float("inf"))
            for label, norm in zip(labels, norms):
                norm_value = float(norm.float().cpu())
                if not torch.isfinite(norm).item():
                    bad_labels.append(f"{label} inf_norm={norm_value}")
                else:
                    largest.append((norm_value, label))

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        largest.sort(reverse=True)
        if rank == 0:
            print(
                "MFSDP_NAN_DIAGNOSTIC "
                f"step={self._nan_diagnostic_step} phase={phase} "
                f"largest={largest[:5]}",
                flush=True,
            )
        if bad_labels:
            print(
                "MFSDP_NAN_DIAGNOSTIC_BAD "
                f"rank={rank} step={self._nan_diagnostic_step} phase={phase} "
                f"entries={bad_labels}",
                flush=True,
            )

        any_bad = torch.tensor(bool(bad_labels), dtype=torch.uint8, device="cuda")
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(any_bad, op=torch.distributed.ReduceOp.MAX)
        if any_bad.item():
            raise FloatingPointError(
                f"M-FSDP diagnostic found a non-finite tensor at step "
                f"{self._nan_diagnostic_step} during {phase}."
            )

    @staticmethod
    def _validate_config(config: OptimizerConfig, model_chunks: List[MegatronModule]) -> None:
        """Validate the MFSDP v2 optimizer support contract."""
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

    def state_dict(self):
        """Return optimizer state.

        MFSDP v2 optimizer checkpointing needs an FSDP-native DTensor state
        contract. Keep this intentionally unsupported for the prototype instead
        of falling back to DDP-buffer assumptions.
        """
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        metadata: Optional[dict] = None,
    ) -> ShardedStateDict:
        """Build a sharded optimizer state dict."""
        raise NotImplementedError("MFSDP v2 optimizer checkpointing is not yet supported.")

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear optimizer-visible sharded grads and any grads filtered from local groups."""
        if not self.is_stub_optimizer:
            self.optimizer.zero_grad(set_to_none=set_to_none)

        # Empty local DTensor shards are filtered out of optimizer param groups
        # as a TE FusedAdam workaround. A rank with no local optimizer params
        # can still have stale module grads to clear.
        for model_chunk in self.model_chunks:
            model_chunk.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        """Step the optimizer, then mark the FSDP execution-trace boundary."""
        result = super().step()
        self.context.complete_trace()
        return result

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

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer and restore MFSDP gradient dtypes."""
        self._nan_diagnostic_step += 1
        self._run_nan_diagnostic("before_optimizer_step")
        success = super().step_with_ready_grads()
        self._run_nan_diagnostic("after_optimizer_and_weight_refresh")
        for parameter, original_grad in self._casted_grads:
            parameter.grad = None
            parameter.grad_dtype = original_grad.dtype
            parameter.grad = original_grad
        self._casted_grads.clear()
        return success

    def _copy_main_params_to_model_params(self) -> None:
        """Refresh MFSDP V2 compute weights after updating optimizer weights."""
        # Walk the model hierarchy instead of the base optimizer's parameter
        # list. Empty local DTensor shards are intentionally omitted from TE
        # FusedAdam, so optimizer parameters can expose FSDP groups in a
        # rank-dependent order. Weight refresh launches collectives and must
        # visit every FSDP group in the same order on every rank. Deliberately
        # do not filter on requires_grad: traversal must remain rank-invariant
        # even when a parameter is locally empty or frozen.
        sync_model_weights_from_main_weights(
            parameter for model_chunk in self.model_chunks for parameter in model_chunk.parameters()
        )

    def _copy_model_params_to_main_params(self, state_dict=None) -> None:
        """No-op: model loads already write into MFSDP v2's main weights."""
