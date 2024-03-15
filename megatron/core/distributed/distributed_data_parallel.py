# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from typing import Dict, Optional

import torch

from .. import parallel_state
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from .grad_buffer import GradBuffer
from .data_parallel_buffer import DataParallelBuffer


class DistributedDataParallel(MegatronModule):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Arguments:
        config: Transformer config object.
        module: Underlying model.
        data_parallel_group: Data-parallel process group.
        accumulate_allreduce_grads_in_fp32: If true, do the gradient accumulation and
            communication in fp32.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        check_for_nan_in_grad: If true, check if local grad norm is NaN.

    """

    def __init__(
        self,
        config: TransformerConfig,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        accumulate_allreduce_grads_in_fp32: bool,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        disable_bucketing: bool = False,
        check_for_nan_in_grad: bool = False,
        bucket_size: int = 40000000,
        zero_stage: int = 0,
        param_dtype: torch.dtype = torch.float32,
        grad_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(config=config)
        self.module = module

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        # Turn off bucketing if overlap_grad_reduce is False, if we are on a pipeline stage
        # that is not the first (since data-parallel communication on these stages is not on
        # the critical path), or if disable_bucketing is True (e.g., we might not want to
        # break up model parameters into buckets for model chunks after the first
        # in the interleaved schedule).
        if not self.overlap_grad_reduce:
            bucket_size = None
        if parallel_state.get_pipeline_model_parallel_rank() > 0:
            bucket_size = None
        if disable_bucketing:
            bucket_size = None

        self.check_for_nan_in_grad = check_for_nan_in_grad
        self.bucket_size = bucket_size
        self.zero_stage = zero_stage
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype

        # Group parameters by their gradient type.
        self.param_to_name = {}
        self.dense_params = []
        self.expert_parallel_params = []
        for name, param in self.module.named_parameters():
            if zero_stage in [0, 1] and not param.requires_grad:
                continue

            if zero_stage in [0, 1]:
                param.grad_added_to_main_grad = False
            self.param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                self.dense_params.append(param)
            else:
                self.expert_parallel_params.append(param)

        if zero_stage in [0, 1]:
            self.allocate_grad_buffer(
                data_parallel_group,
                expert_data_parallel_group,
                accumulate_allreduce_grads_in_fp32,
                bucket_size,
            )
        elif zero_stage == 2:
            self.allocate_data_parallel_buffer_for_zero2(
                data_parallel_group,
                expert_data_parallel_group,
            )
        else:
            raise ValueError(f'Invalid zero_stage: {zero_stage}')

        self.module = module

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param))
                self.grad_accs.append(grad_acc)


    def allocate_grad_buffer(
        self,
        data_parallel_group,
        expert_data_parallel_group,
        accumulate_allreduce_grads_in_fp32,
        bucket_size
    ):
        self.param_to_grad_buffer = {}

        def allocate_grad_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor=1.0,
        ):
            grad_dtype_to_params = {}

            # Group parameters by their gradient type.
            for param in input_params:
                if not param.requires_grad:
                    continue

                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = grad_dtype_to_params.get(dtype, [])
                params.append(param)
                grad_dtype_to_params[dtype] = params

            # Allocate the grad buffers and map the grads.
            grad_buffers = []
            for dtype, params in grad_dtype_to_params.items():
                grad_buffers.append(
                    GradBuffer(
                        dtype,
                        params,
                        data_parallel_group,
                        bucket_size,
                        self.param_to_name,
                        self.overlap_grad_reduce,
                        self.use_distributed_optimizer,
                        gradient_scaling_factor,
                        self.check_for_nan_in_grad,
                    )
                )
                for param in params:
                    self.param_to_grad_buffer[param] = grad_buffers[-1]

            return grad_buffers

        data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)

        # Allocate the grad buffers for dense params' grads.
        self.grad_buffers = allocate_grad_buffers_for_parameters(
            self.dense_params,
            data_parallel_group,
            gradient_scaling_factor=1.0 / data_parallel_world_size,
        )

        # Allocate separate grad buffers for expert parallel params' grads.
        self.expert_parallel_grad_buffers = allocate_grad_buffers_for_parameters(
            self.expert_parallel_params,
            expert_data_parallel_group,
            gradient_scaling_factor=1.0 / data_parallel_world_size,
        )

    def allocate_data_parallel_buffer_for_zero2(
        self,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        # Iterate through parameters in reverse order to roughly follow backprop order.
        self.dense_params = list(reversed(self.dense_params))
        self.expert_parallel_params = list(reversed(self.expert_parallel_params))
        self.dense_param_idx = {param: i for i, param in enumerate(self.dense_params)}
        self.expert_parallel_param_idx = {param: i for i, param in enumerate(self.expert_parallel_params)}

        def allocate_data_parallel_shard_buffer(input_params, dp_group):
            elements = [param.shape for param in input_params]
            data_parallel_rank = torch.distributed.get_rank(dp_group)
            data_parallel_world_size = torch.distributed.get_world_size(dp_group)
            param_buffer = DataParallelBuffer(
                data_parallel_rank=data_parallel_rank,
                data_parallel_world_size=data_parallel_world_size,
                dtype=self.param_dtype,
                elements=elements,
            )
            grad_buffer = DataParallelBuffer(
                data_parallel_rank=data_parallel_rank,
                data_parallel_world_size=data_parallel_world_size,
                dtype=self.grad_dtype,
                elements=elements,
            )
            return param_buffer, grad_buffer

        # Allocate the data parallel buffer's local sharding buffer for both the
        # dense and expert parallel parameters.
        self.dense_param_buffer, self.dense_grad_buffer = allocate_data_parallel_shard_buffer(
            self.dense_params,
            data_parallel_group,
        )
        self.expert_param_buffer, self.expert_grad_buffer = allocate_data_parallel_shard_buffer(
            self.expert_parallel_params,
            expert_data_parallel_group,
        )

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self, param: torch.nn.Parameter,
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def zero_01_param_hook(*unused):
            if param.requires_grad:
                if self.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.overlap_grad_reduce:
                    self.param_to_grad_buffer[param].register_grad_ready(param)

        def zero_2_param_hook(*unused):
            if not param.requires_grad:
                return

            expert_parallel = not getattr(param, 'allreduce', True)
            if expert_parallel:
                idx = self.expert_parallel_param_idx[param]
                self.expert_grad_buffer.put_into_bucket(idx, param.grad.data)
            else:
                idx = self.dense_param_idx[param]
                self.dense_grad_buffer.put_into_bucket(idx, param.grad.data)
            param.grad = None

        if self.zero_stage in [0, 1]:
            return zero_01_param_hook
        elif self.zero_stage == 2:
            return zero_2_param_hook
        else:
            raise ValueError(f'Invalid zero_stage: {self.zero_stage}')

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for grad_buffer in self.grad_buffers + self.expert_parallel_grad_buffers:
            grad_buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for grad_buffer in self.grad_buffers + self.expert_parallel_grad_buffers:
                grad_buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers + self.expert_parallel_grad_buffers:
            grad_buffer.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers + self.expert_parallel_grad_buffers:
            grad_buffer.finish_grad_sync()

    def zero_grad_buffer(self, zero_buffer):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.

        When zero_buffer is set to True, the underlying grad buffer is zeroed out.
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad_added_to_main_grad = False
        for grad_buffer in self.grad_buffers + self.expert_parallel_grad_buffers:
            grad_buffer.reset(zero_buffer)

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                torch.distributed.broadcast(
                    param.data,
                    src=torch.distributed.get_process_group_ranks(self.expert_data_parallel_group),
                    group=self.expert_data_parallel_group,
                )
            else:
                torch.distributed.broadcast(
                    param.data,
                    src=torch.distributed.get_process_group_ranks(self.data_parallel_group),
                    group=self.data_parallel_group,
                )

    def state_dict(self, prefix='', keep_vars=False):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        self.module.load_state_dict(state_dict, strict=strict)
