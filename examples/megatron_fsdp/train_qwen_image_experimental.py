# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Benchmark QwenImage training with the experimental Megatron-FSDP API.

The transformer body is the official Diffusers QwenImage model. Synthetic
packed latents and prompt embeddings keep the example self-contained while
exercising the real flow-matching forward and backward path.
"""

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
    fully_shard_optimizer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlowMatchingBatch:
    """One packed QwenImage flow-matching training batch."""

    model_inputs: dict[str, object]
    target: torch.Tensor


class MemoryHistory:
    """Optionally record CUDA allocation history and write a memory snapshot."""

    def __init__(self, output_dir: str, *, oom_only: bool) -> None:
        self.output_dir = Path(output_dir)
        self.oom_only = oom_only
        self.dumped = False

    def start(self) -> None:
        """Start CUDA memory-history recording."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=200_000, stacks="all")
        if dist.get_rank() == 0:
            logger.info(
                "[rank0] Memory history recording enabled, dump dir=%s oom_only=%s",
                self.output_dir,
                self.oom_only,
            )

    def dump(self, tag: str) -> None:
        """Write this rank's memory snapshot at most once."""
        if self.dumped:
            return
        self.dumped = True
        rank = dist.get_rank()
        path = self.output_dir / f"memory_snapshot_rank{rank}_{tag}.pickle"
        try:
            torch.cuda.memory._dump_snapshot(str(path))
        except Exception as error:  # pylint: disable=broad-exception-caught
            if rank == 0:
                logger.warning("[rank0] Memory snapshot dump failed: %s", error)
            return
        if rank == 0:
            logger.info("[rank0] Memory snapshot dumped: %s", path)

    def stop(self) -> None:
        """Stop CUDA memory-history recording."""
        torch.cuda.memory._record_memory_history(enabled=None)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QwenImage benchmark with experimental Megatron-FSDP."
    )
    parser.add_argument(
        "--backend",
        choices=("mfsdpv2", "fsdp1"),
        default="mfsdpv2",
        help="Distributed backend to benchmark.",
    )
    parser.add_argument(
        "--sharding",
        choices=("full",),
        default="full",
        help="Sharding mode; current experimental Megatron-FSDP supports full sharding.",
    )
    parser.add_argument(
        "--num-gpus-per-node",
        "--num_gpus_per_node",
        dest="num_gpus_per_node",
        type=int,
        default=None,
        help="Optional validation of torchrun LOCAL_WORLD_SIZE.",
    )
    parser.add_argument(
        "--model-id",
        "--pretrained-model-name-or-path",
        "--pretrained_model_name_or_path",
        dest="model_id",
        default="Qwen/Qwen-Image",
        help="Hugging Face model ID or local QwenImage checkpoint directory.",
    )
    parser.add_argument("--subfolder", default="transformer")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=3
    )
    parser.add_argument(
        "--benchmark-steps",
        "--bench-steps",
        "--bench_steps",
        dest="benchmark_steps",
        type=int,
        default=20,
    )
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient_accumulation_steps",
        dest="gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Synthetic image height before VAE downsampling and patch packing.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Synthetic image width before VAE downsampling and patch packing.",
    )
    parser.add_argument(
        "--instruction-sequence-length",
        "--instruction_seq_len",
        dest="instruction_sequence_length",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--vision-patch-size", "--vl_patch_size", dest="vision_patch_size", type=int, default=14
    )
    parser.add_argument(
        "--vision-merge-size", "--vl_merge_size", dest="vision_merge_size", type=int, default=2
    )
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--main-params-dtype",
        choices=("bf16", "fp32"),
        default="bf16",
        help="Megatron-FSDP optimizer-weight dtype; FSDP1 currently requires BF16.",
    )
    parser.add_argument(
        "--attention",
        default="default",
        help="Diffusers attention backend, for example native, flash, or _flash_3.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile every transformer block.")
    parser.add_argument(
        "--gradient-checkpointing",
        "--gradient_checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument("--fused-optimizer", action="store_true")
    parser.add_argument("--use-symm-mem", action="store_true")
    parser.add_argument(
        "--check-convergence",
        "--real-data",
        "--real_data",
        dest="check_convergence",
        action="store_true",
        help="Reuse one fixed synthetic flow-matching batch and require its loss to decrease.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.7,
        help="Require final loss / initial loss to be below this value.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Report global gradient and parameter norms; invalidates benchmark timing.",
    )
    parser.add_argument(
        "--cuda-profiler-capture",
        "--cuda_profiler_capture",
        dest="cuda_profiler_capture",
        action="store_true",
        help="Bracket measured steps with cudaProfilerStart and cudaProfilerStop.",
    )
    parser.add_argument(
        "--record-memory-history",
        metavar="DIRECTORY",
        default=None,
        help="Record CUDA memory history and write per-rank snapshots.",
    )
    parser.add_argument(
        "--record-memory-history-oom-only",
        action="store_true",
        help="Only write memory snapshots after an OOM or CUDA runtime error.",
    )
    return parser.parse_args(argv)


def flat_dp_placements() -> Placements:
    """Return one-dimensional ZeRO-3-style placements."""
    return Placements(dp_axes=["dp"], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def fully_shard_qwen_image_transformer(
    transformer: nn.Module,
    *,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
    use_symm_mem: bool = False,
) -> None:
    """Shard QwenImage blocks bottom-up and shard the transformer root last."""
    transformer_blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(transformer_blocks, (nn.ModuleList, nn.Sequential)):
        raise TypeError(
            "Expected a QwenImage-compatible transformer_blocks ModuleList or Sequential."
        )
    if not transformer_blocks:
        raise ValueError("QwenImage transformer_blocks must not be empty.")

    policy = mixed_precision_policy or MixedPrecisionPolicy()
    for block in transformer_blocks:
        fully_shard(
            block,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=policy,
            use_symm_mem=use_symm_mem,
        )
    fully_shard(
        transformer,
        mesh=mesh,
        placements=placements,
        mixed_precision_policy=policy,
        use_symm_mem=use_symm_mem,
    )


def fully_shard_qwen_image_transformer_fsdp1(
    transformer: nn.Module, *, mesh: DeviceMesh, device: torch.device
) -> nn.Module:
    """Wrap QwenImage with PyTorch FSDP1 using the same per-block units."""
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    transformer_blocks = getattr(transformer, "transformer_blocks", None)
    if not isinstance(transformer_blocks, (nn.ModuleList, nn.Sequential)):
        raise TypeError(
            "Expected a QwenImage-compatible transformer_blocks ModuleList or Sequential."
        )
    if not transformer_blocks:
        raise ValueError("QwenImage transformer_blocks must not be empty.")

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    )
    return FullyShardedDataParallel(
        transformer,
        device_mesh=mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        auto_wrap_policy=ModuleWrapPolicy({type(transformer_blocks[0])}),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=True,
        device_id=device,
        limit_all_gathers=True,
    )


def qwen25vl_vision_tokens(height: int, width: int, *, patch_size: int, merge_size: int) -> int:
    """Estimate Qwen2.5-VL image-token count for the synthetic prompt."""
    merged_patch_size = patch_size * merge_size
    return max(1, round(height / merged_patch_size)) * max(1, round(width / merged_patch_size))


def make_flow_matching_batch(
    transformer: nn.Module,
    *,
    batch_size: int,
    height: int,
    width: int,
    text_sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> FlowMatchingBatch:
    """Create packed latents, conditioning, and the flow velocity target."""
    if height % 16 or width % 16:
        raise ValueError("QwenImage synthetic image height and width must be divisible by 16.")

    grid_height = height // 16
    grid_width = width // 16
    image_sequence_length = grid_height * grid_width
    config = transformer.config
    clean = torch.randn(
        batch_size,
        image_sequence_length,
        config.in_channels,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    noise = torch.randn(clean.shape, device=device, dtype=dtype, generator=generator)
    sigma = torch.rand(batch_size, 1, 1, device=device, dtype=torch.float32, generator=generator)
    noisy = (1.0 - sigma.to(dtype)) * clean + sigma.to(dtype) * noise
    prompt = torch.randn(
        batch_size,
        text_sequence_length,
        config.joint_attention_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return FlowMatchingBatch(
        model_inputs={
            "hidden_states": noisy,
            "encoder_hidden_states": prompt,
            "timestep": sigma.flatten(),
            "img_shapes": [(1, grid_height, grid_width)] * batch_size,
            "return_dict": False,
        },
        target=noise - clean,
    )


def _load_transformer(args: argparse.Namespace, device: torch.device) -> nn.Module:
    """Load the official Diffusers QwenImage transformer."""
    try:
        from diffusers import QwenImageTransformer2DModel
    except ImportError as error:
        raise RuntimeError(
            "This example requires a current Diffusers installation. "
            "See examples/megatron_fsdp/README.md."
        ) from error

    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_id,
        subfolder=args.subfolder,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    return transformer.to(device=device, dtype=torch.bfloat16)


def _load_transformer_one_local_rank_at_a_time(
    args: argparse.Namespace, device: torch.device
) -> nn.Module:
    """Avoid every rank on one node reading checkpoint shards simultaneously."""
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size()))
    transformer = None
    for loader_local_rank in range(local_world_size):
        load_error = None
        if local_rank == loader_local_rank:
            try:
                transformer = _load_transformer(args, device)
            except Exception as error:  # pylint: disable=broad-exception-caught
                load_error = error

        failed = torch.tensor(int(load_error is not None), device=device, dtype=torch.int32)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        if failed.item():
            message = f"QwenImage checkpoint loading failed on local rank {loader_local_rank}."
            if load_error is not None:
                raise RuntimeError(message) from load_error
            raise RuntimeError(message)

    assert transformer is not None
    return transformer


def _initialize_distributed() -> tuple[torch.device, DeviceMesh]:
    """Initialize one NCCL process per local CUDA device."""
    required_environment = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required_environment if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Launch this example with torchrun; missing environment variables: "
            + ", ".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("QwenImage experimental FSDP training requires CUDA.")

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    mesh = init_device_mesh(device.type, (dist.get_world_size(),), mesh_dim_names=("dp",))
    return device, mesh


def _validate_args(args: argparse.Namespace) -> None:
    """Validate arguments before allocating the model."""
    for name in (
        "warmup_steps",
        "benchmark_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "instruction_sequence_length",
    ):
        minimum = 0 if name == "warmup_steps" else 1
        if getattr(args, name) < minimum:
            raise ValueError(f"--{name.replace('_', '-')} must be at least {minimum}.")
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        raise ValueError("--height and --width must be positive multiples of 16.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0 < args.convergence_threshold < 1:
        raise ValueError("--convergence-threshold must be between zero and one.")
    if args.record_memory_history_oom_only and not args.record_memory_history:
        raise ValueError("--record-memory-history-oom-only requires --record-memory-history.")
    if args.num_gpus_per_node is not None:
        if args.num_gpus_per_node < 1:
            raise ValueError("--num-gpus-per-node must be at least one.")
        local_world_size = os.environ.get("LOCAL_WORLD_SIZE")
        if local_world_size is not None and args.num_gpus_per_node != int(local_world_size):
            raise ValueError(
                f"--num-gpus-per-node={args.num_gpus_per_node} does not match "
                f"torchrun LOCAL_WORLD_SIZE={local_world_size}."
            )
    if args.backend == "fsdp1" and args.main_params_dtype != "bf16":
        raise ValueError("--main-params-dtype fp32 is currently supported only by mfsdpv2.")
    if args.backend == "fsdp1" and args.use_symm_mem:
        raise ValueError("--use-symm-mem is supported only by mfsdpv2.")


@contextmanager
def _nvtx_range(name: str) -> Iterator[None]:
    """Annotate one range for Nsight Systems."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _attention_backend(name: str):
    """Return the requested Diffusers attention-backend context."""
    if name == "default":
        return nullcontext()
    try:
        from diffusers.models.attention_dispatch import attention_backend
    except ImportError as error:
        raise RuntimeError(
            "The installed Diffusers version does not expose attention_backend()."
        ) from error
    return attention_backend(name)


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local tensor for a Tensor or DTensor."""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if to_local is not None else tensor


@torch.no_grad()
def _global_gradient_norm(model: nn.Module, device: torch.device) -> tuple[float, int]:
    """Return the global L2 gradient norm and number of visible local gradients."""
    squared_norm = torch.zeros((), device=device, dtype=torch.float64)
    gradient_count = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = _to_local(parameter.grad.detach()).double()
        squared_norm.add_(gradient.square().sum())
        gradient_count += 1
    dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
    return squared_norm.sqrt().item(), gradient_count


@torch.no_grad()
def _global_parameter_stats(model: nn.Module, device: torch.device) -> tuple[float, float, int]:
    """Return global parameter L2 norm, sum, and visible local-view count."""
    squared_norm = torch.zeros((), device=device, dtype=torch.float64)
    total = torch.zeros((), device=device, dtype=torch.float64)
    parameter_count = 0
    for parameter in model.parameters():
        value = _to_local(parameter.detach()).double()
        if value.numel() == 0:
            continue
        squared_norm.add_(value.square().sum())
        total.add_(value.sum())
        parameter_count += 1
    dist.all_reduce(squared_norm, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return squared_norm.sqrt().item(), total.item(), parameter_count


def _mean_across_ranks(value: torch.Tensor) -> float:
    """Return a scalar averaged across ranks."""
    mean = value.detach().float()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    return (mean / dist.get_world_size()).item()


def _format_bytes(byte_count: int) -> str:
    """Format a byte count for benchmark logging."""
    for power, suffix in ((4, "TB"), (3, "GB"), (2, "MB"), (1, "KB")):
        unit = 1024**power
        if byte_count >= unit:
            return f"{byte_count / unit:.2f} {suffix}"
    return f"{byte_count} B"


def _log_memory(tag: str, rank: int) -> None:
    """Log allocator state for this rank."""
    prefix = f"[rank{rank}] {tag}" if tag else f"[rank{rank}]"
    logger.info(
        "%s alloc=%s max_alloc=%s reserved=%s max_reserved=%s",
        prefix,
        _format_bytes(torch.cuda.memory_allocated()),
        _format_bytes(torch.cuda.max_memory_allocated()),
        _format_bytes(torch.cuda.memory_reserved()),
        _format_bytes(torch.cuda.max_memory_reserved()),
    )


def _format_startup_log(
    args: argparse.Namespace, *, world_size: int, text_sequence_length: int
) -> str:
    """Match the original QwenImage benchmark startup log."""
    return (
        f"[{args.backend}] world={world_size} dtype={torch.bfloat16} bs={args.batch_size} "
        f"img={args.height}x{args.width} txt={text_sequence_length} "
        f"sharding={args.sharding} compile={args.compile} gc={args.gradient_checkpointing}"
    )


def _format_step_log(
    args: argparse.Namespace,
    *,
    step: int,
    elapsed: float,
    loss: float,
    verification: tuple[float, int, float, float, int] | None,
) -> str:
    """Match the original QwenImage benchmark per-step log."""
    tag = "warmup" if step < args.warmup_steps else "bench "
    if verification is not None:
        gradient_norm, gradient_count, parameter_norm, parameter_sum, parameter_count = verification
        return (
            f"[{args.backend}] {tag} step {step:3d} | VERIFY (timing invalid) | "
            f"gloss={loss:.6f} | gnorm={gradient_norm:.4f} | n_grad={gradient_count} | "
            f"pnorm={parameter_norm:.8e} | psum={parameter_sum:.8e} | "
            f"n_param={parameter_count}"
        )
    loss_text = f"{loss:.4e}" if args.check_convergence else f"{loss:.4f}"
    return (
        f"[{args.backend}] {tag} step {step:3d} | {elapsed * 1000:8.2f} ms | " f"loss={loss_text}"
    )


def _configure_logging() -> None:
    """Configure this example's logger independently of the root logger."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _build_optimizer(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    """Build AdamW and attach the Megatron-FSDP adapter when required."""
    optimizer_options: dict[str, object]
    if args.fused_optimizer:
        optimizer_options = {"fused": True}
    else:
        optimizer_options = {"foreach": False}
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        **optimizer_options,
    )
    if args.backend == "mfsdpv2":
        fully_shard_optimizer(optimizer)
    return optimizer


def train(args: argparse.Namespace) -> None:
    """Run the QwenImage benchmark and optional fixed-batch convergence check."""
    _validate_args(args)
    device, mesh = _initialize_distributed()
    rank = dist.get_rank()
    memory_history = None
    try:
        torch.manual_seed(args.seed)
        text_sequence_length = args.instruction_sequence_length + qwen25vl_vision_tokens(
            args.height,
            args.width,
            patch_size=args.vision_patch_size,
            merge_size=args.vision_merge_size,
        )
        if rank == 0:
            logger.info(
                _format_startup_log(
                    args,
                    world_size=dist.get_world_size(),
                    text_sequence_length=text_sequence_length,
                )
            )
        transformer = _load_transformer_one_local_rank_at_a_time(args, device)
        transformer.train()
        if args.backend == "mfsdpv2":
            main_params_dtype = (
                torch.bfloat16 if args.main_params_dtype == "bf16" else torch.float32
            )
            fully_shard_qwen_image_transformer(
                transformer,
                mesh=mesh,
                placements=flat_dp_placements(),
                mixed_precision_policy=MixedPrecisionPolicy(
                    main_params_dtype=main_params_dtype,
                    main_grads_dtype=torch.bfloat16,
                    grad_comm_dtype=torch.bfloat16,
                ),
                use_symm_mem=args.use_symm_mem,
            )
            model = transformer
        else:
            model = fully_shard_qwen_image_transformer_fsdp1(transformer, mesh=mesh, device=device)
        if args.compile:
            model_body = model.module if args.backend == "fsdp1" else model
            for block in model_body.transformer_blocks:
                block.compile()

        optimizer = _build_optimizer(model, args)
        generator = torch.Generator(device=device).manual_seed(args.seed + rank)
        fixed_batch = None
        if args.check_convergence:
            fixed_batch = make_flow_matching_batch(
                transformer,
                batch_size=args.batch_size,
                height=args.height,
                width=args.width,
                text_sequence_length=text_sequence_length,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if args.record_memory_history:
            memory_history = MemoryHistory(
                args.record_memory_history, oom_only=args.record_memory_history_oom_only
            )
            memory_history.start()
        _log_memory("after_model_init", rank)

        step_times = []
        initial_loss = None
        final_loss = None
        profiler_active = False
        total_steps = args.warmup_steps + args.benchmark_steps
        with _attention_backend(args.attention):
            for step in range(total_steps):
                if args.cuda_profiler_capture and step == args.warmup_steps:
                    torch.cuda.synchronize()
                    dist.barrier()
                    torch.cuda.profiler.start()
                    profiler_active = True
                    if rank == 0:
                        logger.info("[nsys] cudaProfilerStart")

                step_batch = fixed_batch or make_flow_matching_batch(
                    transformer,
                    batch_size=args.batch_size,
                    height=args.height,
                    width=args.width,
                    text_sequence_length=text_sequence_length,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                torch.cuda.synchronize()
                dist.barrier()
                start_time = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
                forward_time = 0.0
                backward_time = 0.0
                for _ in range(args.gradient_accumulation_steps):
                    forward_start = time.perf_counter()
                    with _nvtx_range("qwenimage_forward"):
                        prediction = model(**step_batch.model_inputs)[0]
                        if prediction.shape != step_batch.target.shape:
                            raise RuntimeError(
                                f"Prediction shape {prediction.shape} does not match "
                                f"flow target {step_batch.target.shape}."
                            )
                        loss = torch.nn.functional.mse_loss(
                            prediction.float(), step_batch.target.float()
                        )
                    forward_time += time.perf_counter() - forward_start
                    backward_start = time.perf_counter()
                    with _nvtx_range("qwenimage_backward"):
                        (loss / args.gradient_accumulation_steps).backward()
                    backward_time += time.perf_counter() - backward_start
                    accumulated_loss.add_(loss.detach())

                if step < args.warmup_steps:
                    _log_memory(
                        f"step={step} fwd_bwd fwd_ms={forward_time * 1000:.1f} "
                        f"bwd_ms={backward_time * 1000:.1f}",
                        rank,
                    )
                gradient_norm = None
                gradient_count = None
                if args.verify:
                    gradient_norm, gradient_count = _global_gradient_norm(model, device)
                with _nvtx_range("optimizer"):
                    optimizer.step()
                parameter_stats = None
                if args.verify:
                    parameter_stats = _global_parameter_stats(model, device)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                measured = step >= args.warmup_steps
                if measured:
                    step_times.append(elapsed)
                mean_loss = _mean_across_ranks(accumulated_loss / args.gradient_accumulation_steps)
                if args.check_convergence and initial_loss is None:
                    initial_loss = mean_loss
                if args.check_convergence:
                    final_loss = mean_loss

                if rank == 0:
                    verification = None
                    if args.verify:
                        parameter_norm, parameter_sum, parameter_count = parameter_stats
                        verification = (
                            gradient_norm,
                            gradient_count,
                            parameter_norm,
                            parameter_sum,
                            parameter_count,
                        )
                    logger.info(
                        _format_step_log(
                            args,
                            step=step,
                            elapsed=elapsed,
                            loss=mean_loss,
                            verification=verification,
                        )
                    )

            if profiler_active:
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.profiler.stop()
                if rank == 0:
                    logger.info("[nsys] cudaProfilerStop")

        peak_memory = torch.tensor(
            torch.cuda.max_memory_allocated(), device=device, dtype=torch.float64
        )
        dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
        average_step_ms = sum(step_times) / len(step_times) * 1000
        if rank == 0:
            logger.info(
                "\n[%s] avg step (n=%d): %.2f ms | peak mem: %.2f GB",
                args.backend,
                len(step_times),
                average_step_ms,
                peak_memory.item() / 1.0e9,
            )

        if args.check_convergence:
            assert initial_loss is not None and final_loss is not None
            loss_ratio = final_loss / max(initial_loss, 1.0e-12)
            if rank == 0:
                logger.info(
                    "[%s] convergence: initial_loss=%.4e final_loss=%.4e ratio=%.3f "
                    "(threshold=%s)",
                    args.backend,
                    initial_loss,
                    final_loss,
                    loss_ratio,
                    args.convergence_threshold,
                )
            if loss_ratio >= args.convergence_threshold:
                raise AssertionError(
                    f"Convergence failed: loss ratio {loss_ratio:.4f} is not below "
                    f"{args.convergence_threshold:.4f}."
                )

        _log_memory("final", rank)
        if memory_history is not None and not memory_history.oom_only:
            memory_history.dump("final")
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if memory_history is not None:
            memory_history.dump("error")
        raise
    finally:
        if memory_history is not None:
            memory_history.stop()
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    """Run the QwenImage experimental FSDP benchmark."""
    _configure_logging()
    train(parse_args())


if __name__ == "__main__":
    main()
