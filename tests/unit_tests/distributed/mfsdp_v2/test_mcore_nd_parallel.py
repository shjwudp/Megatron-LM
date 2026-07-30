# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import copy
import time

import pytest
import torch
from torch.testing import assert_close

import megatron.core.parallel_state as mpu
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import HAVE_TE_MXFP8TENSOR
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.utils import is_torch_min_version
from megatron.training.global_vars import destroy_global_vars
from tests.unit_tests.distributed.mfsdp_v1.utils import (
    make_gpt_mock_data_iterator,
    make_moe_args_model_and_optimizer,
    pretrain_forward_backward,
    set_manual_seed,
)
from tests.unit_tests.test_utilities import Utils

STRICT_LOSS_ATOL = 5e-3
STRICT_PARAM_ATOL = 5e-3
STRICT_PARAM_RTOL = 1e-3


@pytest.fixture(scope="class")
def ref_cache():
    """
    Shared read/write cache for an class.
    Keys: arbitrary strings, values: anything (tensors, dicts, etc.).
    """
    return {}


class TestMegatronFSDPE2E:
    def teardown_method(self):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    @staticmethod
    def _normalize_param_name(name):
        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    @staticmethod
    def _materialize_param_tensor(param):
        from torch.distributed.tensor import DTensor

        from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
            uneven_dtensor_to_full_tensor,
        )
        from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor

        tensor = param.detach()
        if isinstance(tensor, DTensor):
            tensor = uneven_dtensor_to_full_tensor(tensor)
        elif is_float8tensor(tensor):
            tensor = dequantize_fp8_tensor(tensor)
        return tensor.detach().float().cpu()

    @staticmethod
    def _capture_named_params(model_chunks):
        snapshots = {}
        for chunk_idx, model_chunk in enumerate(model_chunks):
            for name, param in model_chunk.named_parameters():
                tensor = TestMegatronFSDPE2E._materialize_param_tensor(param)
                if torch.distributed.get_rank() == 0:
                    key = f"{chunk_idx}.{TestMegatronFSDPE2E._normalize_param_name(name)}"
                    snapshots[key] = tensor
        return snapshots

    @staticmethod
    def _training_loop(seed=42, **kwargs):
        VOCAB_SIZE = kwargs.pop("vocab_size", 100)
        MAX_SEQ_LEN = kwargs.pop("seq_length", 128)
        MICRO_BATCH_SIZE = kwargs.pop("micro_batch_size", 2)
        GLOBAL_BATCH_SIZE = kwargs.pop("global_batch_size", 32)
        NUM_TRAINING_STEPS = kwargs.pop("train_iters", 20)
        TP = kwargs.pop("TP", 1)
        PP = kwargs.pop("PP", 1)
        VPP = kwargs.pop("VPP", None)
        EP = kwargs.pop("EP", 1)
        ETP = kwargs.pop("ETP", 1)
        OUTER_DP = kwargs.pop("OUTER_DP", 1)
        capture_param_snapshots = kwargs.pop("capture_param_snapshots", False)
        return_dict = kwargs.pop("return_dict", capture_param_snapshots)

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            expert_model_parallel_size=EP,
            expert_tensor_parallel_size=ETP,
            num_distributed_optimizer_instances=OUTER_DP,
        )
        DP_GROUP = mpu.get_data_parallel_group()

        set_manual_seed(seed)

        model_chunks, optim = make_moe_args_model_and_optimizer(
            ut_filename="test_mcore_fully_sharded_data_parallel.py",
            micro_batch_size=MICRO_BATCH_SIZE,
            global_batch_size=GLOBAL_BATCH_SIZE,
            vocab_size=VOCAB_SIZE,
            padded_vocab_size=VOCAB_SIZE,
            seq_length=MAX_SEQ_LEN,
            sequence_parallel=TP > 1,
            expert_model_parallel_size=EP,
            tensor_model_parallel_size=TP,
            pipeline_model_parallel_size=PP,
            num_layers_per_virtual_pipeline_stage=VPP,
            train_iters=NUM_TRAINING_STEPS,
            **kwargs,
        )

        data_iterator = make_gpt_mock_data_iterator(
            dp_group=DP_GROUP,
            vocab_size=VOCAB_SIZE,
            sequence_length=MAX_SEQ_LEN,
            batch_size=MICRO_BATCH_SIZE,
            num_samples=GLOBAL_BATCH_SIZE * NUM_TRAINING_STEPS,
        )

        outputs = []
        param_snapshots = []

        for step in range(NUM_TRAINING_STEPS):
            t0 = time.time()
            optim.zero_grad()
            output = pretrain_forward_backward(
                model=model_chunks,
                data_iterator=data_iterator,
                sequence_length=MAX_SEQ_LEN,
                micro_batch_size=MICRO_BATCH_SIZE,
                num_micro_batches=GLOBAL_BATCH_SIZE // MICRO_BATCH_SIZE // DP_GROUP.size(),
            )
            optim.step()

            outputs.append(output[-1])
            if torch.distributed.get_rank() == 0:
                elapsed = time.time() - t0
                mem_alloc = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
                print(
                    f"[Step {step + 1}/{NUM_TRAINING_STEPS}] "
                    f"loss={output[-1]['lm loss'].item():.6f} "
                    f"time={elapsed:.2f}s "
                    f"mem_alloc={mem_alloc:.2f}GiB "
                    f"mem_reserved_max={mem_reserved:.2f}GiB"
                )
                torch.cuda.reset_peak_memory_stats()
            if capture_param_snapshots:
                param_snapshots.append(TestMegatronFSDPE2E._capture_named_params(model_chunks))

        Utils.destroy_model_parallel()

        if return_dict:
            result = {"outputs": outputs}
            if capture_param_snapshots:
                result["param_snapshots"] = param_snapshots
            return result
        return outputs

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Test needs to be updated for torch >= 2.4.0"
    )
    @pytest.mark.parametrize("nd_topology", [pytest.param({"EP": 2}, id="EP2")])
    @pytest.mark.parametrize(
        "spec_configs",
        [
            pytest.param(
                dict(
                    data_parallel_sharding_strategy="optim_grads_params",
                    recompute_granularity="full",
                    recompute_method="uniform",
                    overlap_param_gather=True,
                    overlap_grad_reduce=True,
                    use_megatron_fsdp=True,
                    gradient_accumulation_fusion=True,
                ),
                id="optim_grads_params_double_buffer",
            ),
            pytest.param(
                dict(
                    bf16=True,
                    data_parallel_sharding_strategy="optim_grads_params",
                    fp8="e4m3",
                    fp8_param_gather=True,
                    fp8_recipe="mxfp8",
                    moe_grouped_gemm=True,
                    overlap_param_gather=True,
                    overlap_grad_reduce=True,
                    use_megatron_fsdp=True,
                ),
                id="optim_grads_params_mxfp8_param_gather",
            ),
            pytest.param(
                dict(
                    bf16=True,
                    data_parallel_sharding_strategy="optim_grads_params",
                    fp8="e4m3",
                    fp8_param_gather=True,
                    fp8_recipe="mxfp8",
                    moe_grouped_gemm=True,
                    use_megatron_fsdp=True,
                    moe_token_dispatcher_type="alltoall",
                    overlap_moe_expert_parallel_comm=True,
                    delay_wgrad_compute=True,
                ),
                id="ep_overlap-optim_grads_params",
            ),
        ],
    )
    def test_compatible_with_nd_parallel(self, ref_cache, nd_topology, spec_configs):
        if spec_configs.get("fp8_recipe") == "mxfp8" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 10
            or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")

        if spec_configs.get("overlap_moe_expert_parallel_comm"):
            from megatron.core.utils import is_te_min_version

            if not is_te_min_version("2.3.0"):
                pytest.skip("EP overlap requires Transformer Engine >= 2.3.0")

        reference_kind = "distopt"
        ref_cache_key = (
            reference_kind,
            tuple(sorted(nd_topology.items())),
            tuple(sorted((key, repr(value)) for key, value in spec_configs.items())),
        )
        if ref_cache_key not in ref_cache:
            reference_spec_configs = copy.deepcopy(spec_configs)
            reference_spec_configs["use_megatron_fsdp"] = False
            reference_spec_configs["gradient_accumulation_fusion"] = False
            reference_spec_configs["fp8_param_gather"] = False
            ref_cache[ref_cache_key] = TestMegatronFSDPE2E._training_loop(
                use_distributed_optimizer=True, **nd_topology, **reference_spec_configs
            )

        fsdp_spec_configs = copy.deepcopy(spec_configs)
        fsdp_spec_configs.setdefault("gradient_accumulation_fusion", False)
        outputs = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            init_model_with_meta_device=True,
            ckpt_format="fsdp_dtensor",
            **nd_topology,
            **fsdp_spec_configs,
        )
        reference_outputs = ref_cache[ref_cache_key]

        if torch.distributed.get_rank() == 0:
            for step, (output, ref_output) in enumerate(zip(outputs, reference_outputs)):
                loss = output["lm loss"]
                ref_loss = ref_output["lm loss"]
                assert_close(
                    loss,
                    ref_loss,
                    atol=0,
                    rtol=0.05,
                    msg=(
                        f"Loss mismatch at step {step}, FSDP Loss = {loss.detach().item()}, "
                        f"Reference Loss = {ref_loss.detach().item()}"
                        f", Compare = {compare_losses(loss.detach().item(), ref_loss.detach().item())}"
                        f", outputs = {outputs}, reference_outputs = {reference_outputs}"
                    ),
                )

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Test needs to be updated for torch >= 2.4.0"
    )
    @pytest.mark.parametrize(
        "case",
        [
            pytest.param(
                dict(
                    strategy="optim",
                    precision_configs=dict(bf16=True),
                    reference_kind="distopt",
                    capture_param_snapshots=True,
                ),
                id="bf16-optim",
            ),
            pytest.param(
                dict(
                    strategy="optim_grads",
                    precision_configs=dict(bf16=True),
                    reference_kind="distopt",
                    capture_param_snapshots=True,
                ),
                id="bf16-optim_grads",
            ),
            pytest.param(
                dict(
                    strategy="optim_grads_params",
                    precision_configs=dict(bf16=True),
                    reference_kind="distopt",
                    capture_param_snapshots=True,
                ),
                id="bf16-optim_grads_params",
            ),
            pytest.param(
                dict(
                    strategy="optim_grads_params",
                    precision_configs=dict(
                        bf16=True,
                        fp8="e4m3",
                        fp8_param_gather=True,
                        fp8_recipe="mxfp8",
                        main_grads_dtype="fp32",
                        main_params_dtype="fp32",
                        exp_avg_dtype="bf16",
                        exp_avg_sq_dtype="bf16",
                        moe_grouped_gemm=True,
                        use_precision_aware_optimizer=True,
                    ),
                    reference_kind="distopt",
                    capture_param_snapshots=False,
                ),
                id="mxfp8_param_gather-optim_grads_params",
            ),
        ],
    )
    def test_strict_iter_equivalence_zero_strategies(self, ref_cache, case):
        strategy = case["strategy"]
        precision_configs = case["precision_configs"]
        if precision_configs.get("fp8_recipe") == "mxfp8" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 10
            or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")
        if Utils.world_size < 2:
            pytest.skip("Requires at least 2 distributed ranks for ZeRO sharding")

        common_configs = dict(
            data_parallel_sharding_strategy=strategy,
            train_iters=3,
            seq_length=64,
            micro_batch_size=1,
            global_batch_size=8,
            init_model_with_meta_device=False,
            gradient_accumulation_fusion=False,
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            **precision_configs,
        )
        reference_kind = case["reference_kind"]
        capture_param_snapshots = case["capture_param_snapshots"]
        ref_cache_key = (
            "strict_iter_equivalence",
            reference_kind,
            strategy,
            capture_param_snapshots,
            tuple(sorted((key, repr(value)) for key, value in common_configs.items())),
        )

        if ref_cache_key not in ref_cache:
            ref_cache[ref_cache_key] = TestMegatronFSDPE2E._training_loop(
                use_distributed_optimizer=True,
                capture_param_snapshots=capture_param_snapshots,
                return_dict=True,
                **common_configs,
            )

        fsdp_configs = copy.deepcopy(common_configs)
        fsdp_configs["use_megatron_fsdp"] = True
        actual = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            ckpt_format="fsdp_dtensor",
            capture_param_snapshots=capture_param_snapshots,
            return_dict=True,
            **fsdp_configs,
        )
        reference = ref_cache[ref_cache_key]

        if torch.distributed.get_rank() != 0:
            return

        assert len(actual["outputs"]) == len(reference["outputs"])
        for step, (output, ref_output) in enumerate(zip(actual["outputs"], reference["outputs"])):
            loss = output["lm loss"]
            ref_loss = ref_output["lm loss"]
            assert_close(
                loss,
                ref_loss,
                atol=STRICT_LOSS_ATOL,
                rtol=0,
                msg=(
                    f"Loss mismatch at step {step}, strategy={strategy}, "
                    f"actual={loss.detach().item()}, reference={ref_loss.detach().item()}, "
                    f"compare={compare_losses(loss.detach().item(), ref_loss.detach().item())}"
                ),
            )

        if not capture_param_snapshots:
            return

        assert len(actual["param_snapshots"]) == len(reference["param_snapshots"])
        for step, (params, ref_params) in enumerate(
            zip(actual["param_snapshots"], reference["param_snapshots"])
        ):
            missing = sorted(set(ref_params) ^ set(params))
            assert (
                not missing
            ), f"Parameter key mismatch at step {step}, strategy={strategy}: {missing[:20]}"
            for name in sorted(ref_params):
                assert_close(
                    params[name],
                    ref_params[name],
                    atol=STRICT_PARAM_ATOL,
                    rtol=STRICT_PARAM_RTOL,
                    msg=(
                        f"Parameter mismatch at step {step}, strategy={strategy}, "
                        f"name={name}, actual_shape={tuple(params[name].shape)}, "
                        f"reference_shape={tuple(ref_params[name].shape)}"
                    ),
                )

    @pytest.mark.skipif(
        not is_torch_min_version("2.4.0"), reason="Test needs to be updated for torch >= 2.4.0"
    )
    @pytest.mark.parametrize(
        "strategy,precision_configs",
        [
            pytest.param(
                strategy,
                dict(
                    bf16=True,
                    fp8="e4m3",
                    fp8_param_gather=True,
                    fp8_recipe="mxfp8",
                    main_grads_dtype="fp32",
                    main_params_dtype="fp32",
                    exp_avg_dtype="bf16",
                    exp_avg_sq_dtype="bf16",
                    moe_grouped_gemm=True,
                    use_precision_aware_optimizer=True,
                ),
                id=f"mxfp8_param_gather-{strategy}",
            )
            for strategy in ("optim", "optim_grads")
        ],
    )
    def test_zero_strategy_non_equivalent_precision_paths_run(self, strategy, precision_configs):
        if precision_configs.get("fp8_recipe") == "mxfp8" and (
            not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 10
            or not HAVE_TE_MXFP8TENSOR
        ):
            pytest.skip("Requires PyTorch & CUDA device with TE MXFP8Tensor support")
        if Utils.world_size < 2:
            pytest.skip("Requires at least 2 distributed ranks for ZeRO sharding")

        outputs = TestMegatronFSDPE2E._training_loop(
            use_megatron_fsdp=True,
            ckpt_format="fsdp_dtensor",
            data_parallel_sharding_strategy=strategy,
            train_iters=3,
            seq_length=64,
            micro_batch_size=1,
            global_batch_size=8,
            init_model_with_meta_device=False,
            gradient_accumulation_fusion=False,
            overlap_param_gather=False,
            overlap_grad_reduce=False,
            **precision_configs,
        )

        if torch.distributed.get_rank() != 0:
            return

        assert len(outputs) == 3
        for step, output in enumerate(outputs):
            loss = output["lm loss"]
            assert torch.isfinite(loss), (
                f"Non-finite loss at step {step}, strategy={strategy}, "
                f"precision={precision_configs}"
            )


def compare_losses(loss_a: float, loss_b: float, reference: str = "b"):
    abs_diff = abs(loss_a - loss_b)

    if reference == "a":
        ref = loss_a
    else:
        ref = loss_b

    if ref == 0:
        rel_diff = float("inf")
    else:
        rel_diff = abs_diff / ref

    if loss_a < loss_b:
        better = "a"
    elif loss_b < loss_a:
        better = "b"
    else:
        better = "equal"

    return {"abs_diff": abs_diff, "rel_diff": rel_diff, "better": better}
