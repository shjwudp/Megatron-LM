# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore optimizer wrapper for experimental Megatron-FSDP v2."""

from typing import Callable, List, Optional

import torch
from torch.distributed.tensor import DTensor

from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..dist_checkpointing.mapping import ShardedStateDict
from ..distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    sync_model_weights_from_main_weights,
)
from ..transformer.module import MegatronModule
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer
from .optimizer_config import OptimizerConfig

_GRAD_CAST_ALIGNMENT_BYTES = 256
_OPTIMIZER_GRAD_CAST_ARENA = "optimizer_grad_cast"


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
        self._casted_grads: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
        self._grad_cast_pool_keys: list[tuple[object, ...]] = []

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
        assert not self._grad_cast_pool_keys
        grads_to_cast = []
        for parameter in self.get_parameters():
            if parameter.grad is None:
                continue
            if parameter.grad.dtype == parameter.data.dtype:
                continue

            original_grad = parameter.grad
            local_grad = (
                original_grad.to_local() if isinstance(original_grad, DTensor) else original_grad
            )
            grads_to_cast.append((parameter, original_grad, local_grad))

        allocator = self.context.trace_pool_allocator
        if allocator is None or allocator.use_symmetric_memory:
            # Symmetric-memory allocations must have the same sizes and order on
            # every rank. Empty optimizer shards make that untrue for this local
            # cast workspace, so retain the ordinary allocator in that mode.
            for parameter, original_grad, _local_grad in grads_to_cast:
                parameter.grad = None
                parameter.grad_dtype = parameter.data.dtype
                parameter.grad = original_grad.to(dtype=parameter.data.dtype)
                self._casted_grads.append((parameter, original_grad))
            return

        groups: dict[
            tuple[torch.device, torch.dtype],
            list[tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]],
        ] = {}
        for parameter, original_grad, local_grad in grads_to_cast:
            group_key = (local_grad.device, parameter.data.dtype)
            groups.setdefault(group_key, []).append((parameter, original_grad, local_grad))

        try:
            for (device, dtype), entries in groups.items():
                element_size = torch.empty((), dtype=dtype).element_size()
                alignment = max(1, _GRAD_CAST_ALIGNMENT_BYTES // element_size)
                offsets = []
                total_numel = 0
                for _parameter, _original_grad, local_grad in entries:
                    total_numel = ((total_numel + alignment - 1) // alignment) * alignment
                    offsets.append(total_numel)
                    total_numel += local_grad.numel()

                allocation_key = (id(self), "optimizer_grad_cast", device, dtype)
                local_buffer = allocator.allocate(
                    allocation_key,
                    total_numel,
                    dtype,
                    device,
                    arena=_OPTIMIZER_GRAD_CAST_ARENA,
                )
                self._grad_cast_pool_keys.append(allocation_key)

                for (parameter, original_grad, local_grad), offset in zip(entries, offsets):
                    casted_local_grad = local_buffer.narrow(
                        0, offset, local_grad.numel()
                    ).view(local_grad.shape)
                    casted_local_grad.copy_(local_grad)
                    if isinstance(original_grad, DTensor):
                        casted_grad = DTensor.from_local(
                            local_tensor=casted_local_grad,
                            device_mesh=original_grad.device_mesh,
                            placements=original_grad.placements,
                            run_check=False,
                            shape=original_grad.shape,
                            stride=casted_local_grad.stride(),
                        )
                    else:
                        casted_grad = casted_local_grad

                    parameter.grad = None
                    parameter.grad_dtype = parameter.data.dtype
                    parameter.grad = casted_grad
                    self._casted_grads.append((parameter, original_grad))
        except Exception:
            self._restore_model_grads()
            raise

    def _restore_model_grads(self) -> None:
        """Restore original gradient views and release pooled cast storage."""
        for parameter, original_grad in self._casted_grads:
            parameter.grad = None
            parameter.grad_dtype = original_grad.dtype
            parameter.grad = original_grad
        self._casted_grads.clear()

        allocator = self.context.trace_pool_allocator
        if allocator is not None:
            for allocation_key in self._grad_cast_pool_keys:
                allocator.free(allocation_key)
        self._grad_cast_pool_keys.clear()

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer and restore MFSDP gradient dtypes."""
        try:
            return super().step_with_ready_grads()
        finally:
            self._restore_model_grads()

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
