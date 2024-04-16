# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from contextlib import contextmanager
from typing import Dict, Optional

import torch

from .. import parallel_state
from ..transformer.module import MegatronModule
from ..transformer.transformer_config import TransformerConfig
from .data_parallel_buffer import DataParallelBuffer
from .param_and_grad_buffer import ParamAndGradBuffer


class DistributedDataParallel(MegatronModule):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
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
        overlap_param_gather: bool,
        use_distributed_optimizer: bool,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        disable_bucketing: bool = False,
        check_for_nan_in_grad: bool = False,
        bucket_size: int = 40000000,
        data_parallel_sharding_strategy: str = "NO_OP",
        in_optimizer_param_dtype: torch.dtype = torch.float32,
        grad_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(config=config)
        self.module = module

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.overlap_param_gather = overlap_param_gather
        self.use_distributed_optimizer = use_distributed_optimizer
        self.param_all_gather_handler_map = {}

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
        self.data_parallel_sharding_strategy = data_parallel_sharding_strategy
        self.in_optimizer_param_dtype = in_optimizer_param_dtype
        self.grad_dtype = grad_dtype

        self.module = module
        self.param_to_buffer = {}

        # Group parameters by their gradient type.
        self.param_to_name = {}
        self.dense_params = []
        self.expert_parallel_params = []
        for name, param in self.module.named_parameters():
            if (
                data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]
                and not param.requires_grad
            ):
                continue

            if data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
                param.grad_added_to_main_grad = False
            self.param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                self.dense_params.append(param)
            else:
                self.expert_parallel_params.append(param)

        if data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
            self.allocate_grad_buffer(
                data_parallel_group,
                expert_data_parallel_group,
                accumulate_allreduce_grads_in_fp32,
                bucket_size,
            )
        elif data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS":
            self.allocate_data_parallel_buffer(
                data_parallel_group, expert_data_parallel_group,
            )
        else:
            raise ValueError(
                f'Invalid data_parallel_sharding_strategy: {data_parallel_sharding_strategy}'
            )

        self.module = module
        self._named_parameter_shardings_cache = None

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.use_distributed_optimizer:

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        self.register_forward_hook()

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.register_backward_hook()

    def register_backward_hook(self):
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
        bucket_size,
    ):
        self.param_to_grad_buffer = {}

        def allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor=1.0,
        ):
            param_and_grad_dtype_to_params = {}

            # Group parameters by their gradient type.
            for param in input_params:
                if not param.requires_grad:
                    continue

                param_dtype = param.dtype
                grad_dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

            # Allocate the grad buffers and map the grads.
            buffers = []
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    ParamAndGradBuffer(
                        param_dtype,
                        grad_dtype,
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
                    self.param_to_buffer[param] = buffers[-1]

            return buffers

        data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers = allocate_buffers_for_parameters(
            self.dense_params,
            data_parallel_group,
            gradient_scaling_factor=1.0 / data_parallel_world_size,
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers = allocate_buffers_for_parameters(
            self.expert_parallel_params,
            expert_data_parallel_group,
            gradient_scaling_factor=1.0 / data_parallel_world_size,
        )

    def allocate_data_parallel_buffer(
        self,
        data_parallel_group: torch.distributed.ProcessGroup,
        expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        # Iterate through parameters in reverse order to roughly follow backprop order.
        self.dense_params = list(reversed(self.dense_params))
        self.expert_parallel_params = list(reversed(self.expert_parallel_params))

        def allocate_data_parallel_shard_buffer(input_params, dp_group):
            device = (
                input_params[0].device if len(input_params) > 0 else torch.cuda.current_device()
            )
            param_buffer = DataParallelBuffer(
                data_parallel_group=dp_group,
                dtype=self.in_optimizer_param_dtype,
                device=device,
                parameters=input_params,
            )
            for i, param in enumerate(input_params):
                param_buffer.set_item(i, param.data)

            grad_buffer = DataParallelBuffer(
                data_parallel_group=dp_group,
                dtype=self.grad_dtype,
                device=device,
                parameters=input_params,
            )
            return param_buffer, grad_buffer

        # Allocate the data parallel buffer's local sharding buffer for both the
        # dense and expert parallel parameters.
        self.dense_param_buffer, self.dense_grad_buffer = allocate_data_parallel_shard_buffer(
            self.dense_params, data_parallel_group,
        )
        self.expert_param_buffer, self.expert_grad_buffer = allocate_data_parallel_shard_buffer(
            self.expert_parallel_params, expert_data_parallel_group,
        )

    def named_parameter_shardings(self):
        if self.data_parallel_sharding_strategy != "OPTIMIZER_STATES_AND_GRADS":
            return None

        if self._named_parameter_shardings_cache:
            return self._named_parameter_shardings_cache

        def _named_parameter_shardings(params, param_buffer, grad_buffer):
            named_param_shardings = []
            for item_id, param in enumerate(params):
                param_shard = param_buffer.get_item(item_id)
                if param_shard.numel() == 0:
                    # parameter not in this rank
                    continue

                def closure(param_shard, grad_shard, param):
                    def reset_attribute():
                        setattr(param_shard, 'grad', grad_shard)
                        setattr(param_shard, 'requires_grad', param.requires_grad)
                        if hasattr(param, 'sequence_parallel'):
                            setattr(param_shard, 'sequence_parallel', param.sequence_parallel)

                    return reset_attribute

                setattr(
                    param_shard,
                    'reset_attribute',
                    closure(param_shard, grad_buffer.get_item(item_id), param),
                )
                param_shard.reset_attribute()
                named_param_shardings.append((self.param_to_name[param], param_shard))

            return named_param_shardings

        self._named_parameter_shardings_cache = _named_parameter_shardings(
            self.dense_params, self.dense_param_buffer, self.dense_grad_buffer,
        ) + _named_parameter_shardings(
            self.expert_parallel_params, self.expert_param_buffer, self.expert_grad_buffer,
        )

        return self._named_parameter_shardings_cache

    def register_forward_hook(self):
        """
        Registers forward hooks to be called before and after the forward pass.
        """

        def distog_forward_pre_hook_closure():
            def forward_pre_hook(module, input):
                for param in module.parameters(recurse=False):
                    handler = self.param_all_gather_handler_map[param]
                    if handler:
                        handler.wait()
                    del self.param_all_gather_handler_map[param]

            return forward_pre_hook

        if (
            self.overlap_param_gather
            and self.data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS"
        ):
            self.module.register_forward_pre_hook(distog_forward_pre_hook_closure())

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        if (
            self.data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS"
            and self.is_first_microbatch
            and self.module.training
        ):
            self.model_parameters_allgather()

        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self, param: torch.nn.Parameter,
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def grad_buffer_param_hook(*unused):
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
                    self.param_to_buffer[param].register_grad_ready(param)

        def dp_buffer_param_hook(*unused):
            if not param.requires_grad:
                return

            def gradient_accumulate(grad_buffer, param):
                item_idx = grad_buffer.param_idx[param]
                bucket = grad_buffer.put_into_bucket(item_idx, param.grad.data)
                if len(bucket.items) == len(bucket.requires_grad_items):
                    grad_buffer.wait_for_previous_reduce_scatter()
                    grad_buffer.reduce_scatter_bucket_and_add_on_local_shard(
                        bucket.id, async_op=self.overlap_grad_reduce
                    )

            expert_parallel = not getattr(param, 'allreduce', True)
            grad_buffer = self.expert_grad_buffer if expert_parallel else self.dense_grad_buffer
            gradient_accumulate(grad_buffer, param)
            param.grad = None

        if self.data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
            return grad_buffer_param_hook
        elif self.data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS":
            return dp_buffer_param_hook
        else:
            raise ValueError(
                f'Invalid data_parallel_sharding_strategy: {self.data_parallel_sharding_strategy}'
            )

    @torch.no_grad()
    def model_parameters_allgather(self):
        def _params_allgather(param_buffer, param_dtype):
            for i, shard_bucket_index in enumerate(param_buffer.shard_bucket_index_map):
                local_data_index = shard_bucket_index.local_data_index
                offset = shard_bucket_index.bucket_data_index
                shard_size = shard_bucket_index.size

                # do parameter all-gather in temporary buffer
                bucket = param_buffer.allocate_bucket(i, param_dtype)
                bucket.data[offset : offset + shard_size] = param_buffer.data[
                    local_data_index : local_data_index + shard_size
                ]
                all_gather_handler = torch.distributed.all_gather_into_tensor(
                    output_tensor=bucket.data,
                    input_tensor=bucket.data[offset : offset + shard_size],
                    group=param_buffer.data_parallel_group,
                    async_op=self.overlap_param_gather,
                )

                bucket_index = param_buffer.bucket_index_map[i]
                for item_index in bucket_index.items:
                    param = param_buffer.parameters[item_index.item_id]

                    self.param_all_gather_handler_map[param] = all_gather_handler

                    # copy bucket data back to local parameters
                    item = param_buffer.get_item_from_bucket(bucket, item_index.item_id)
                    param.copy_(item.view_as(param))

        dense_param_dtype = self.dense_params[0].dtype if len(self.dense_params) else torch.float32
        expert_param_dtype = (
            self.expert_parallel_params[0].dtype
            if len(self.expert_parallel_params)
            else torch.float32
        )
        _params_allgather(self.dense_param_buffer, dense_param_dtype)
        _params_allgather(self.expert_param_buffer, expert_param_dtype)

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        if self.data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.is_last_microbatch = False
        try:
            yield
        finally:
            if self.data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
                for buffer in self.buffers + self.expert_parallel_buffers:
                    buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if self.data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.start_grad_sync()
        elif self.data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS":
            pass

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        if self.data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.finish_grad_sync()
        elif self.data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS":
            # reset the attribute of the parameters when the grad sync is finished
            for _, param_shard in self.named_parameter_shardings():
                param_shard.reset_attribute()

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        if self.data_parallel_sharding_strategy in ["NO_OP", "OPTIMIZER_STATES"]:
            for param in self.module.parameters():
                if param.requires_grad:
                    param.grad_added_to_main_grad = False
            for buffer in self.buffers + self.expert_parallel_buffers:
                buffer.reset()
        elif self.data_parallel_sharding_strategy == "OPTIMIZER_STATES_AND_GRADS":
            # Zero out the grad buffers.
            self.dense_grad_buffer.data.zero_()
            self.expert_grad_buffer.data.zero_()

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
                self.expert_param_buffer.set_item(
                    item_id=self.expert_param_buffer.param_idx[param], item_data=param.data,
                )
            else:
                torch.distributed.broadcast(
                    param.data,
                    src=torch.distributed.get_process_group_ranks(self.data_parallel_group),
                    group=self.data_parallel_group,
                )
                self.dense_param_buffer.set_item(
                    item_id=self.dense_param_buffer.param_idx[param], item_data=param.data,
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
