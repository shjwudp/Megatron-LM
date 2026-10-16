# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Numerical coverage of the production VPP and combined 1F1B paths."""

from contextlib import nullcontext

import pytest
import torch
from torch import nn

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.combined_1f1b import combined_1f1b_schedule_for_no_pipelining
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.a2a_overlap.utils import (
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    forward_step_func,
    get_test_config,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils


def _ddp_config():
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        data_parallel_sharding_strategy="optim_grads_params",
    )


class _Chunk(nn.Module):
    def __init__(self, config, chunk_id):
        super().__init__()
        self.config = config
        self.vp_stage = chunk_id
        self.model_type = ModelType.encoder_or_decoder
        self.fc1 = nn.Linear(16, 16, bias=False)
        self.fc2 = nn.Linear(16, 16, bias=False)
        self.input_tensor = None

    def set_input_tensor(self, tensors):
        self.input_tensor = tensors[0]

    def forward(self, x):
        if self.input_tensor is not None:
            x = self.input_tensor
        return self.fc2(torch.tanh(self.fc1(x)))


def test_vpp_training_validation_training(distributed_setup):
    """PP2/VPP2: grouped microbatches and validation match an unsharded run."""
    if distributed_setup.world_size < 4:
        pytest.skip("Requires PP2 and at least DP2.")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        config = TransformerConfig(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=1,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            microbatch_group_size_per_vp_stage=2,
        )
        torch.manual_seed(1234 + torch.distributed.get_rank(pg.pp))
        reference = [_Chunk(config, i).cuda() for i in range(2)]
        chunks = [_Chunk(config, i).cuda() for i in range(2)]
        for chunk, ref in zip(chunks, reference):
            chunk.load_state_dict(ref.state_dict())
        with fully_shard_context(device=distributed_setup.device, caller_managed_grad_sync=True):
            sharded = [
                FullyShardedDataParallelV2(
                    config, _ddp_config(), chunk, fsdp_unit_modules=[nn.Linear], pg_collection=pg
                )
                for chunk in chunks
            ]
        optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in sharded]
        for optimizer in optimizers:
            fully_shard_optimizer(optimizer)
        ref_optimizers = [torch.optim.SGD(model.parameters(), lr=0.02) for model in reference]
        torch.manual_seed(4321 + torch.distributed.get_rank(pg.dp_cp))
        inputs = [torch.randn(4, 2, 1, 16, device="cuda") for _ in range(2)]

        def step(iterator, model):
            output = model(next(iterator))

            def loss(output):
                value = output.square().mean()
                return value, {"loss": value.detach()}

            return output, loss

        for forward_only in (False, True, False):
            results = []
            for models, opts, is_sharded in (
                (reference, ref_optimizers, False),
                (sharded, optimizers, True),
            ):
                config.no_sync_func = [model.no_sync for model in models] if is_sharded else None
                for optimizer in opts:
                    optimizer.zero_grad()
                context = torch.no_grad() if forward_only else nullcontext()
                with context:
                    losses = forward_backward_pipelining_with_interleaving(
                        forward_step_func=step,
                        data_iterator=[iter(x) for x in inputs],
                        model=models,
                        num_microbatches=4,
                        seq_length=2,
                        micro_batch_size=1,
                        forward_only=forward_only,
                        p2p_communicator=P2PCommunicator(pg.pp, config),
                        pg_collection=pg,
                    )
                results.append(losses)
                if not forward_only:
                    if not is_sharded:
                        for model in models:
                            for param in model.parameters():
                                torch.distributed.all_reduce(param.grad, group=pg.dp_cp)
                                param.grad.div_(pg.dp_cp.size())
                    if is_sharded:
                        for model in models:
                            model.finish_grad_sync()
                    for optimizer in opts:
                        optimizer.step()
            for expected, actual in zip(*results):
                torch.testing.assert_close(actual["loss"], expected["loss"])
        # Also check parameters on non-last PP ranks, which do not return losses.
        for ref, model in zip(reference, sharded):
            for expected, actual in zip(ref.parameters(), model.parameters()):
                torch.testing.assert_close(actual.full_tensor(), expected)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("delay_wgrad", [False, True])
def test_combined_1f1b_training_matches_eager(distributed_setup, delay_wgrad):
    """Real MoE plans overlap three microbatches, including delayed TE wgrad."""
    if distributed_setup.world_size < 8:
        pytest.skip("Requires EP4 with expert DP2.")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=4
    )
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        with deterministic_mode():
            common = {
                "gradient_accumulation_fusion": False,
                "moe_token_dispatcher_type": "alltoall",
            }
            ref_config = get_test_config(num_layers=2, extra_kwargs=common)
            reference = build_gpt_model(ref_config, vocab_size=128)
            initial = reset_model(reference)
            config = get_test_config(
                num_layers=2,
                extra_kwargs={
                    **common,
                    "overlap_moe_expert_parallel_comm": True,
                    "delay_wgrad_compute": delay_wgrad,
                },
            )
            model = build_gpt_model(config, vocab_size=128)
            reset_model(model, initial)
            reference = FullyShardedDataParallelV2(
                ref_config, _ddp_config(), reference, pg_collection=pg
            )
            model = FullyShardedDataParallelV2(config, _ddp_config(), model, pg_collection=pg)
            ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            fully_shard_optimizer(ref_optimizer)
            fully_shard_optimizer(optimizer)
            data = [build_input_data(seq_len=16, vocab_size=128) for _ in range(3)]
            for _ in range(3):
                ref_optimizer.zero_grad()
                expected = []
                for batch in data:
                    output = reference(**batch).float().sum()
                    # The scheduler logs the same loss tensor it scales for backward.
                    expected.append(output.detach() / len(data))
                    (output / len(data)).backward()
                reference.finish_grad_sync()
                ref_optimizer.step()
                optimizer.zero_grad()
                actual = []
                combined_1f1b_schedule_for_no_pipelining(
                    forward_step_func=forward_step_func,
                    data_iterator=iter(data),
                    model=model,
                    num_microbatches=len(data),
                    input_tensor=None,
                    output_tensor_grad=None,
                    forward_data_store=actual,
                    config=config,
                    collect_non_loss_data=False,
                    first_val_step=None,
                    forward_only=False,
                    no_sync_func=model.no_sync,
                    total_num_tokens=torch.zeros([], dtype=torch.int, device="cuda"),
                    check_first_val_step=lambda value: value,
                )
                model.finish_grad_sync()
                optimizer.step()
                torch.testing.assert_close(
                    torch.stack([loss["lm loss"] for loss in actual]),
                    torch.stack(expected),
                    rtol=0.005,
                    atol=0.05,
                )
            for expected, actual in zip(reference.parameters(), model.parameters()):
                torch.testing.assert_close(
                    actual.to_local(), expected.to_local(), rtol=0.01, atol=0.001
                )
    finally:
        PipelineOffloadManager.reset_instance()
        Utils.destroy_model_parallel()
