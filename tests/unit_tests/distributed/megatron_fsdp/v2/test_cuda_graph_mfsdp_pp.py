# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pipeline-parallel regression for M-FSDP recompute CUDA Graphs."""

import os
from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard
from megatron.core.enums import ModelType
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection


@pytest.fixture(scope="module", autouse=True)
def distributed_environment():
    """Initialize the four-rank NCCL test environment."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
    yield


class _PipelineStage(nn.Module):
    """Two sharded layers owned by one pipeline stage."""

    def __init__(self, checkpoint_fn):
        """Create a stage using the supplied checkpoint callable.

        :param checkpoint_fn: Native checkpoint callable.
        :type checkpoint_fn: Callable
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)])
        self.checkpoint_fn = checkpoint_fn
        self.input_tensor = None
        self.model_type = ModelType.encoder_or_decoder
        self.config = ModelParallelConfig(
            pipeline_model_parallel_size=2, pipeline_dtype=torch.float32, sequence_parallel=False
        )
        self.config.hidden_size = 4

    def set_input_tensor(self, input_tensor):
        """Store the activation received from the previous stage.

        :param input_tensor: Pipeline input list.
        :type input_tensor: list
        """
        self.input_tensor = input_tensor

    def _checkpointed_forward(self, value):
        """Run both captured layers.

        :param value: Stage input.
        :type value: torch.Tensor
        :return: Stage output.
        :rtype: torch.Tensor
        """
        for layer in self.layers:
            value = layer(value)
        return value

    def forward(self, value=None):
        """Run the checkpointed stage.

        :param value: First-stage input, defaults to the received activation.
        :type value: Optional[torch.Tensor]
        :return: Stage output.
        :rtype: torch.Tensor
        """
        if value is None and self.input_tensor is not None:
            value = self.input_tensor[0]
        return self.checkpoint_fn(self._checkpointed_forward, value)


@pytest.mark.parametrize("enable_cuda_graph", (False, True))
def test_mfsdp_recompute_real_pp2_dp2_1f1b(request, enable_cuda_graph):
    """Run the real non-interleaved schedule with two in-flight microbatches.

    :param request: Pytest request used for graph-pool cleanup.
    :type request: pytest.FixtureRequest
    :param enable_cuda_graph: Whether to enable recompute CUDA Graphs.
    :type enable_cuda_graph: bool
    """
    if torch.distributed.get_world_size() != 4:
        pytest.skip("This integration test requires PP2 x DP2 on four ranks")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    pp_group, _ = torch.distributed.new_subgroups_by_enumeration(((0, 2), (1, 3)))
    dp_group, _ = torch.distributed.new_subgroups_by_enumeration(((0, 1), (2, 3)))
    tp_group, _ = torch.distributed.new_subgroups_by_enumeration(((0,), (1,), (2,), (3,)))
    dp_ranks = torch.distributed.get_process_group_ranks(dp_group)
    dp_mesh = DeviceMesh.from_group(
        [tp_group, dp_group], device_type="cuda", mesh=[dp_ranks], mesh_dim_names=("dp_outer", "dp")
    )
    checkpoint_fn = partial(checkpoint, use_reentrant=False)
    model = _PipelineStage(checkpoint_fn).to(device)
    model.config.batch_p2p_sync = False
    with torch.no_grad():
        for layer in model.layers:
            layer.weight.copy_(torch.eye(4, device=device))
    for layer in model.layers:
        fully_shard(
            layer,
            mesh=dp_mesh,
            sharding_strategy="optim_grads_params",
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_activation_recompute=enable_cuda_graph,
            cuda_graph_max_pending_forwards=2 if enable_cuda_graph else 1,
        )
    fully_shard(
        model,
        mesh=dp_mesh,
        sharding_strategy="optim_grads_params",
        enable_unshard_prefetch=False,
        enable_async_reduce_grad=False,
    )
    model.config.no_sync_func = model.no_sync
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    communicator = P2PCommunicator(pp_group, model.config)
    groups = ProcessGroupCollection()
    groups.tp = tp_group
    groups.pp = pp_group
    groups.cp = tp_group
    groups.dp_cp = dp_group
    groups.tp_dp_cp = dp_group
    groups.embd = None
    groups.pos_embd = None

    def release_graph_pool():
        """Release captured graph state."""
        if not enable_cuda_graph:
            return
        if any(layer._fsdp_cg_pending_backwards for layer in model.layers):
            model.release_pending()
        model.release_memory_pool()
        torch.cuda.synchronize()

    request.addfinalizer(release_graph_pool)

    def forward_step(data_iterator, stage):
        """Run one pipeline forward step.

        :param data_iterator: First-stage input iterator.
        :type data_iterator: Iterator[torch.Tensor]
        :param stage: Local pipeline stage.
        :type stage: nn.Module
        :return: Output tensor and loss callback.
        :rtype: tuple
        """
        value = next(data_iterator) if communicator.is_pp_first_stage else None
        output = stage(value)

        def loss_func(tensor):
            """Compute the scalar test loss."""
            loss = tensor.float().square().mean()
            return loss, {"loss": loss.detach()}

        return output, loss_func

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        inputs = iter(
            (
                torch.full((2, 2, 4), 1.0 + step, device=device),
                torch.full((2, 2, 4), 2.0 + step, device=device),
            )
        )
        before = tuple(parameter.to_local().detach().clone() for parameter in model.parameters())
        forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step,
            data_iterator=inputs,
            model=model,
            num_microbatches=2,
            seq_length=2,
            micro_batch_size=2,
            forward_only=False,
            p2p_communicator=communicator,
            pg_collection=groups,
        )
        model.finish_grad_sync()
        optimizer.step()
        torch.distributed.barrier(device_ids=[device.index])
        torch.cuda.synchronize()

        if enable_cuda_graph:
            assert all(layer._fsdp_cg_pending_backwards == 0 for layer in model.layers)
        assert any(
            not torch.equal(parameter.to_local(), previous)
            for parameter, previous in zip(model.parameters(), before)
        )
        if enable_cuda_graph and step == 2:
            runner = model._fsdp_root_context.cuda_graph_runner
            assert runner._captured
            assert runner._ordered_replay_events
