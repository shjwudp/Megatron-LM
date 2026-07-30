# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MCore adapter and optimizer integration tests for experimental MFSDP v2."""

from dataclasses import replace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.fully_sharded_optimizer import FullyShardedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


def _build_layer(config: TransformerConfig) -> TransformerLayer:
    return TransformerLayer(
        config=config,
        submodules=get_gpt_layer_local_spec().submodules,
        layer_number=1,
        add_layer_offset=False,
    )


def _build_block(config: TransformerConfig) -> TransformerBlock:
    return TransformerBlock(config=config, spec=get_gpt_layer_local_spec()).to(
        device="cuda", dtype=config.params_dtype
    )


class _PipelineStage(torch.nn.Module):
    """Minimal pipeline stage that accepts activations from the 1F1B schedule."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear = torch.nn.Linear(
            config.hidden_size,
            config.hidden_size,
            device="cuda",
            dtype=config.params_dtype,
        )
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        """Save the activation supplied by the pipeline schedule."""
        self.input_tensor = input_tensor[0]

    def forward(self, hidden_states=None):
        """Run this stage on local input or the activation received from its peer."""
        if hidden_states is None:
            hidden_states = self.input_tensor
        return self.linear(hidden_states)


def _pipeline_forward_step(data_iterator, model):
    hidden_states = next(data_iterator) if parallel_state.is_pipeline_first_stage() else None
    output = model(hidden_states)

    def loss_func(output_tensor):
        loss = output_tensor.float().square().mean()
        return loss, {"loss": loss.detach()}

    return output, loss_func


class TestMcoreAdapter:
    """Exercise a dense MCore transformer block over two data-parallel ranks."""

    def setup_method(self):
        Utils.initialize_model_parallel(1, 1)
        if torch.distributed.get_world_size() < 2:
            pytest.skip("MFSDP v2 MCore integration test requires at least two ranks.")
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_wraps_fsdp_unit_modules_before_root(self):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        layer = _build_layer(config)
        model = torch.nn.Sequential(layer, torch.nn.Linear(config.hidden_size, config.hidden_size))
        model = model.to(device="cuda", dtype=config.params_dtype)

        wrapped = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.float32,
                fsdp_all_gather_in_start_param_sync=False,
            ),
            module=model,
            fsdp_unit_modules=[TransformerLayer],
            pg_collection=self.pg_collection,
        )

        assert isinstance(wrapped.module, FsdpModule)
        assert isinstance(wrapped.module[0], FsdpModule)

        # Post-order wrapping gives the selected TransformerLayer its own parameter group;
        # the root FSDP unit should own only the parameters of the remaining Linear module.
        child_parameter_names = {
            name for group in wrapped.module[0].parameter_groups for name in group.parameter_names
        }
        root_parameter_names = {
            name for group in wrapped.module.parameter_groups for name in group.parameter_names
        }
        assert child_parameter_names
        assert root_parameter_names == {"1.weight", "1.bias"}

    def test_build_train_and_step(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        reference_model = _build_block(config)
        model = _build_block(config)
        model.load_state_dict(reference_model.state_dict())
        # get_megatron_optimizer expects every model chunk to expose ddp_config. The
        # reference model remains unwrapped/unsharded, so it cannot use the
        # DistributedOptimizer path that expects DDP/FSDP buffer metadata.
        reference_model.ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)

        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.bfloat16,
                fsdp_all_gather_in_start_param_sync=False,
            ),
            module=model,
            pg_collection=self.pg_collection,
        )

        reference_optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_distributed_optimizer=False,
            clip_grad=0.0,
        )
        reference_optimizer = get_megatron_optimizer(
            reference_optimizer_config, [reference_model], use_gloo_process_groups=False
        )
        optimizer_config = replace(reference_optimizer_config)
        with pytest.raises(
            ValueError, match="MFSDP v2 currently requires use_distributed_optimizer=False"
        ):
            get_megatron_optimizer(
                replace(reference_optimizer_config, use_distributed_optimizer=True),
                [model],
                use_gloo_process_groups=False,
            )
        optimizer = get_megatron_optimizer(optimizer_config, [model], use_gloo_process_groups=False)
        assert isinstance(optimizer, FullyShardedOptimizer)
        optimizer.reload_model_params()

        steps = [
            [
                torch.randn(8, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
                for _ in range(2)
            ]
            for _ in range(3)
        ]

        reference_losses = []
        for microbatches in steps:
            reference_optimizer.zero_grad(set_to_none=True)
            microbatch_losses = []
            for batch in microbatches:
                reference_output = reference_model(hidden_states=batch, attention_mask=None)
                reference_loss = reference_output.square().mean()
                (reference_loss / len(microbatches)).backward()
                microbatch_losses.append(reference_loss.detach())
            reference_success, _, _ = reference_optimizer.step()
            assert reference_success
            reference_losses.append(torch.stack(microbatch_losses).mean())

        losses = []
        for microbatches in steps:
            model.zero_grad_buffer()
            optimizer.zero_grad(set_to_none=True)
            microbatch_losses = []
            for batch in microbatches:
                output = model(hidden_states=batch, attention_mask=None)
                loss = output.square().mean()
                (loss / len(microbatches)).backward()
                microbatch_losses.append(loss.detach())
            success, _, _ = optimizer.step()
            assert success
            losses.append(torch.stack(microbatch_losses).mean())

        losses = torch.stack(losses)
        reference_losses = torch.stack(reference_losses)
        assert torch.isfinite(losses).all()
        assert torch.isfinite(reference_losses).all()
        torch.testing.assert_close(losses, reference_losses, rtol=1e-3, atol=0)


class TestMcoreAdapterPipelineParallel:
    """Exercise MFSDP v2 with the non-interleaved pipeline schedule."""

    def setup_method(self):
        if Utils.world_size < 4:
            pytest.skip("MFSDP v2 PP test requires PP=2 and at least two data-parallel ranks.")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=2
        )
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model_parallel_cuda_manual_seed(1234)

    def teardown_method(self):
        Utils.destroy_model_parallel()

    def test_1f1b_train_step_uses_stage_local_fsdp_mesh(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=4,
            ffn_hidden_size=32,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.bfloat16,
                fsdp_all_gather_in_start_param_sync=False,
            ),
            module=_PipelineStage(config),
            pg_collection=self.pg_collection,
        )

        expected_mesh_ranks = torch.distributed.get_process_group_ranks(
            self.pg_collection.dp_cp
        )
        parameter_groups = model.module.parameter_groups
        assert parameter_groups
        for parameter_group in parameter_groups:
            assert parameter_group.mesh.mesh.flatten().tolist() == expected_mesh_ranks
        assert len(expected_mesh_ranks) == torch.distributed.get_world_size() // 2
        assert self.pg_collection.pp.size() == 2

        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=1.0e-3,
                weight_decay=0.0,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=False,
                clip_grad=0.0,
            ),
            [model],
            use_gloo_process_groups=False,
        )
        assert isinstance(optimizer, FullyShardedOptimizer)
        optimizer.reload_model_params()
        optimizer.zero_grad(set_to_none=True)

        num_microbatches = 4
        seq_length = 8
        micro_batch_size = 2
        microbatches = iter(
            [
                torch.randn(
                    seq_length,
                    micro_batch_size,
                    config.hidden_size,
                    device="cuda",
                    dtype=config.params_dtype,
                )
                for _ in range(num_microbatches)
            ]
        )
        losses = get_forward_backward_func()(
            forward_step_func=_pipeline_forward_step,
            data_iterator=microbatches,
            model=[model],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

        for parameter_group in parameter_groups:
            assert parameter_group.main_grad is not None
            assert torch.isfinite(parameter_group.main_grad.local_buffer).all()
        success, _, _ = optimizer.step()
        assert success
        if parallel_state.is_pipeline_last_stage():
            assert len(losses) == num_microbatches
            assert all(torch.isfinite(loss["loss"]) for loss in losses)
        else:
            assert not losses
