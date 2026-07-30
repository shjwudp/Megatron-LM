# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import fully_shard_optimizer
from megatron.core.pipeline_parallel.combined_1f1b import combined_1f1b_schedule_for_no_pipelining
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerLayer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    assert_models_equal,
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    forward_step_func,
    get_test_config,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
VOCAB_SIZE = 128
LR = 0.01


class TestExperimentalFSDP1F1BOverlap:
    """Compare experimental MFSDP v2 combined 1F1B with standard backward."""

    def setup_method(self):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self):
        Utils.destroy_model_parallel()

    @staticmethod
    def _make_ddp_config():
        return DistributedDataParallelConfig(
            use_megatron_fsdp=True,
            megatron_fsdp_version=2,
            use_distributed_optimizer=False,
            data_parallel_sharding_strategy="optim_grads_params",
            megatron_fsdp_main_params_dtype=torch.float32,
            megatron_fsdp_main_grads_dtype=torch.bfloat16,
            fsdp_all_gather_in_start_param_sync=False,
        )

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    def test_two_microbatch_steady_state(self):
        """Exercise warmup, one overlapped steady phase, and cooldown."""
        with deterministic_mode():
            microbatches = [
                build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE) for _ in range(2)
            ]
            reference_config = get_test_config(num_layers=2)
            reference_model = build_gpt_model(reference_config, vocab_size=VOCAB_SIZE)
            initial_parameters = reset_model(reference_model)
            reference_fsdp = FullyShardedDataParallel(
                config=reference_config,
                ddp_config=self._make_ddp_config(),
                module=reference_model,
                fsdp_unit_modules=[TransformerLayer],
                pg_collection=self.pg_collection,
            )
            reference_optimizer = torch.optim.SGD(reference_fsdp.parameters(), lr=LR)
            fully_shard_optimizer(reference_optimizer)

            overlap_config = get_test_config(
                num_layers=2, extra_kwargs={"overlap_moe_expert_parallel_comm": True}
            )
            overlap_model = build_gpt_model(overlap_config, vocab_size=VOCAB_SIZE)
            reset_model(overlap_model, initial_parameters)
            overlap_fsdp = FullyShardedDataParallel(
                config=overlap_config,
                ddp_config=self._make_ddp_config(),
                module=overlap_model,
                fsdp_unit_modules=[TransformerLayer],
                pg_collection=self.pg_collection,
            )
            overlap_optimizer = torch.optim.SGD(overlap_fsdp.parameters(), lr=LR)
            fully_shard_optimizer(overlap_optimizer)

            reference_optimizer.zero_grad()
            reference_losses = []
            for data in microbatches:
                loss = float16_to_fp32(reference_fsdp(**data)).sum()
                loss.backward()
                reference_losses.append(loss.detach())
            reference_optimizer.step()

            overlap_optimizer.zero_grad()
            forward_data_store = []
            combined_1f1b_schedule_for_no_pipelining(
                forward_step_func=forward_step_func,
                data_iterator=iter(microbatches),
                model=overlap_fsdp,
                num_microbatches=len(microbatches),
                input_tensor=None,
                output_tensor_grad=None,
                forward_data_store=forward_data_store,
                config=overlap_config,
                collect_non_loss_data=False,
                first_val_step=None,
                forward_only=False,
                no_sync_func=nullcontext,
                total_num_tokens=torch.zeros([], dtype=torch.int, device="cuda"),
                check_first_val_step=lambda condition: condition,
            )
            torch.cuda.synchronize()
            overlap_losses = [entry["lm loss"] for entry in forward_data_store]
            overlap_optimizer.step()

            torch.testing.assert_close(
                torch.stack(overlap_losses),
                torch.stack(reference_losses),
                rtol=1e-2,
                atol=0,
            )
            assert_models_equal(reference_fsdp, overlap_fsdp)
