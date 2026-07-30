# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import gc

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    fully_shard_optimizer as fully_shard_optimizer_v2,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_optimizer
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    apply_flex_backend_kwargs,
    assert_models_equal,
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    fsdp_train_step,
    get_test_config,
    get_valid_dispatcher_configs,
    get_valid_fp8_flags,
    overlap_train_step,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

# Transformer Engine 2.17 aborts in the A2A overlap suite with a pybind11 GIL dec_ref failure.
pytestmark = pytest.mark.flaky_in_dev

SEQ_LEN = 32
VOCAB_SIZE = 128
NUM_STEPS = 3
LR = 0.01


class TestFSDP1F1BOverlap:
    """Verify FSDP integration with the 1F1B overlap schedule.

    Uses combined_1f1b_schedule_for_no_pipelining (the production code path)
    to drive the overlap schedule, then compares per-step loss and final
    weights against standard FSDP forward/backward.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()
        self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize("sharding_strategy", ["optim_grads_params", "optim_grads"])
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    def test_fsdp_1f1b_training_step(
        self,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        sharding_strategy,
        shared_expert_intermediate_size,
    ):
        self._run_test_helper(
            dispatcher_type,
            fp8_flag,
            sharding_strategy,
            shared_expert_intermediate_size,
            flex_backend=flex_backend,
        )

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize(
        "recompute_modules",
        [[], ["core_attn", "mla_up_proj", "layernorm", "moe_act", "mlp", "shared_experts"]],
    )
    @pytest.mark.parametrize(
        "offload_modules",
        [[], ["attn_norm", "core_attn", "attn_proj", "mlp_norm", "expert_fc1", "moe_act"]],
    )
    def test_fsdp_1f1b_memory_opt(self, recompute_modules, offload_modules):
        self._run_test_helper(
            dispatcher_type="alltoall",
            sharding_strategy="optim_grads_params",
            recompute_modules=recompute_modules,
            offload_modules=offload_modules,
        )

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    def test_experimental_fsdp_1f1b_two_microbatches(self):
        """Exercise a real warmup/steady/cooldown combined-1F1B sequence."""
        from contextlib import nullcontext

        from megatron.core.pipeline_parallel.combined_1f1b import (
            combined_1f1b_schedule_for_no_pipelining,
        )
        from megatron.core.transformer.module import float16_to_fp32
        from tests.unit_tests.a2a_overlap.utils import forward_step_func

        def make_ddp_config():
            return DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                megatron_fsdp_version=2,
                use_distributed_optimizer=False,
                data_parallel_sharding_strategy="optim_grads_params",
                megatron_fsdp_main_params_dtype=torch.float32,
                megatron_fsdp_main_grads_dtype=torch.bfloat16,
                fsdp_all_gather_in_start_param_sync=False,
            )

        with deterministic_mode():
            microbatches = [
                build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE) for _ in range(2)
            ]
            reference_config = get_test_config(num_layers=2)
            reference_model = build_gpt_model(reference_config, vocab_size=VOCAB_SIZE)
            initial_parameters = reset_model(reference_model)
            reference_fsdp = FullyShardedDataParallel(
                config=reference_config,
                ddp_config=make_ddp_config(),
                module=reference_model,
                fsdp_unit_modules=[TransformerLayer],
                pg_collection=self.pg_collection,
            )
            reference_optimizer = torch.optim.SGD(reference_fsdp.parameters(), lr=LR)
            fully_shard_optimizer_v2(reference_optimizer)

            overlap_config = get_test_config(
                num_layers=2, extra_kwargs={"overlap_moe_expert_parallel_comm": True}
            )
            overlap_model = build_gpt_model(overlap_config, vocab_size=VOCAB_SIZE)
            reset_model(overlap_model, initial_parameters)
            overlap_fsdp = FullyShardedDataParallel(
                config=overlap_config,
                ddp_config=make_ddp_config(),
                module=overlap_model,
                fsdp_unit_modules=[TransformerLayer],
                pg_collection=self.pg_collection,
            )
            overlap_optimizer = torch.optim.SGD(overlap_fsdp.parameters(), lr=LR)
            fully_shard_optimizer_v2(overlap_optimizer)

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

    def _run_test_helper(
        self,
        dispatcher_type="alltoall",
        fp8_flag=None,
        sharding_strategy="optim_grads_params",
        shared_expert_intermediate_size=None,
        recompute_modules=None,
        offload_modules=None,
        flex_backend=None,
    ):
        """Verify multi-step FSDP training with overlap produces identical
        per-step loss and final weights as standard FSDP training.

        Covers FP8 recipes, activation recomputation, shared experts, and
        different token dispatcher types.  Reference uses standard
        forward/backward; test uses combined_1f1b_schedule_for_no_pipelining.
        """
        num_layers = 2
        extra_kwargs = {}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size
        if recompute_modules:
            extra_kwargs["recompute_granularity"] = "selective"
            extra_kwargs["recompute_modules"] = recompute_modules
        if offload_modules:
            extra_kwargs["fine_grained_activation_offloading"] = True
            extra_kwargs["offload_modules"] = offload_modules

        def _make_ddp_config():
            return DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                data_parallel_sharding_strategy=sharding_strategy,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                megatron_fsdp_main_params_dtype=None,
            )

        with deterministic_mode():
            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)

            # --- Reference: FSDP model with standard training loop ---
            ref_config = get_test_config(num_layers=num_layers, extra_kwargs=extra_kwargs)
            ref_model = build_gpt_model(ref_config, vocab_size=VOCAB_SIZE)
            init_params = reset_model(ref_model)

            ref_fsdp = FullyShardedDataParallel(
                config=ref_config,
                ddp_config=_make_ddp_config(),
                module=ref_model,
                fsdp_unit_modules=[TransformerLayer],
            )
            ref_opt = torch.optim.SGD(ref_fsdp.parameters(), lr=LR)
            ref_opt = fully_shard_optimizer(optimizer=ref_opt)

            # --- Test: FSDP model with overlap training loop ---
            test_kwargs = {**extra_kwargs, "overlap_moe_expert_parallel_comm": True}
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=test_kwargs)
            test_model = build_gpt_model(test_config, vocab_size=VOCAB_SIZE)
            reset_model(test_model, init_params)

            test_fsdp = FullyShardedDataParallel(
                config=test_config,
                ddp_config=_make_ddp_config(),
                module=test_model,
                fsdp_unit_modules=[TransformerLayer],
            )
            test_opt = torch.optim.SGD(test_fsdp.parameters(), lr=LR)
            test_opt = fully_shard_optimizer(optimizer=test_opt)

            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                if hasattr(ref_fsdp, 'set_is_first_microbatch'):
                    ref_fsdp.set_is_first_microbatch()
                ref_loss = fsdp_train_step(ref_fsdp, ref_opt, data)
                test_loss = overlap_train_step(test_fsdp, test_opt, test_config, data)

                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            assert_models_equal(ref_fsdp, test_fsdp)

            del ref_fsdp, test_fsdp, ref_opt, test_opt
            gc.collect()
            torch.cuda.empty_cache()
