# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Train a toy MLP with experimental Megatron-FSDP or PyTorch FSDP2."""

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Partial,
    Placements,
    Replicate,
    fully_shard,
    fully_shard_optimizer,
    microbatch,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy

logger = logging.getLogger(__name__)


class ToyBlock(nn.Module):
    """Gated MLP block similar to a transformer feed-forward layer."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.use_activation_checkpointing = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the block, optionally under non-reentrant activation checkpointing."""
        if self.use_activation_checkpointing:
            return checkpoint(self._forward_impl, inputs, use_reentrant=False)
        return self._forward_impl(inputs)

    def _forward_impl(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down(torch.nn.functional.gelu(self.gate(inputs)) * self.up(inputs))
        )


class ToyModel(nn.Module):
    """Stack of toy gated MLP blocks."""

    def __init__(self, dim: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(ToyBlock(dim) for _ in range(num_layers))
        self.out = nn.Linear(dim, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the toy model."""
        for layer in self.layers:
            inputs = layer(inputs)
        return self.out(inputs)

    def enable_activation_checkpointing(self) -> None:
        """Checkpoint every repeated block."""
        for layer in self.layers:
            layer.use_activation_checkpointing = True


class TeacherStudentData:
    """Deterministic teacher/student regression data for convergence checks."""

    def __init__(
        self,
        *,
        dim: int,
        num_layers: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.device = device
        self.dtype = dtype
        set_seed(seed)
        self.teacher = ToyModel(dim=dim, num_layers=num_layers).to(device=device, dtype=dtype)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    @torch.no_grad()
    def sample(
        self, *, batch_size: int, sequence_length: int, step: int, rank: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a distinct reproducible sample for one step and DP rank."""
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed + 1_000_003 * step + 9_176 * rank)
        inputs = torch.randn(
            batch_size,
            sequence_length,
            self.dim,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        return inputs, self.teacher(inputs)


class AppState(Stateful):
    """DCP-compatible model and optimizer application state."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, object]:
        """Return canonical distributed model and optimizer state dictionaries."""
        model_state, optimizer_state = get_state_dict(self.model, self.optimizer)
        return {"model": model_state, "optimizer": optimizer_state}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore canonical distributed model and optimizer state dictionaries."""
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Toy experimental Megatron-FSDP training")
    parser.add_argument("--backend", choices=("mfsdpv2", "fsdp2"), default="mfsdpv2")
    parser.add_argument(
        "--use-megatron-fsdp",
        dest="backend",
        action="store_const",
        const="mfsdpv2",
        help="Compatibility alias selecting --backend mfsdpv2.",
    )
    parser.add_argument("--model-dim", type=int, default=1024)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--activation-checkpoint", action="store_true")
    parser.add_argument("--enable-hsdp", action="store_true")
    parser.add_argument("--use-symm-mem", action="store_true")
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--ckpt-interval", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--record-memory-history", metavar="DIRECTORY", default=None)
    parser.add_argument("--use-real-data", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--convergence-threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Seed CPU and CUDA RNGs so every rank initializes identically."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def initialize_distributed(enable_hsdp: bool) -> tuple[torch.device, DeviceMesh]:
    """Initialize NCCL and a one- or two-dimensional data-parallel mesh."""
    if not torch.cuda.is_available():
        raise RuntimeError("The toy experimental FSDP example requires CUDA.")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if enable_hsdp:
        if world_size < 4 or world_size % 2:
            raise ValueError("HSDP requires an even world size of at least four.")
        mesh = init_device_mesh(
            device.type, (2, world_size // 2), mesh_dim_names=("dp_outer", "dp_inner")
        )
    else:
        mesh = init_device_mesh(device.type, (world_size,), mesh_dim_names=("dp",))
    return device, mesh


def toy_placements(mesh: DeviceMesh) -> Placements:
    """Return full-shard placements, including outer optimizer sharding for HSDP."""
    if mesh.ndim == 1:
        return Placements(dp_axes=["dp"], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])
    return Placements(
        dp_axes=["dp_outer", "dp_inner"],
        parameter=[Replicate(), Flat()],
        gradient=[Partial(dist.ReduceOp.AVG), Flat()],
        optimizer=[Flat(), Flat()],
    )


def fully_shard_toy_model(
    model: ToyModel, *, mesh: DeviceMesh, backend: str, use_symm_mem: bool = False
) -> nn.Module:
    """Shard each toy block bottom-up and then shard the model root."""
    if backend == "mfsdpv2":
        policy = MixedPrecisionPolicy(
            main_params_dtype=torch.bfloat16,
            main_grads_dtype=torch.bfloat16,
            grad_comm_dtype=torch.bfloat16,
        )
        placements = toy_placements(mesh)
        for layer in model.layers:
            fully_shard(
                layer,
                mesh=mesh,
                placements=placements,
                mixed_precision_policy=policy,
                use_symm_mem=use_symm_mem,
            )
        fully_shard(
            model,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=policy,
            use_symm_mem=use_symm_mem,
        )
        return model

    if backend != "fsdp2":
        raise ValueError(f"Unknown backend: {backend}")
    if use_symm_mem:
        raise ValueError("--use-symm-mem is supported only by mfsdpv2.")
    try:
        from torch.distributed.fsdp import fully_shard as fully_shard_fsdp2
    except ImportError:
        # PyTorch exposed FSDP2 from the composable namespace before promoting
        # fully_shard to torch.distributed.fsdp.
        from torch.distributed._composable.fsdp import fully_shard as fully_shard_fsdp2

    for layer in model.layers:
        fully_shard_fsdp2(layer, mesh=mesh)
    fully_shard_fsdp2(model, mesh=mesh)
    return model


def build_optimizer(
    model: nn.Module, *, backend: str, learning_rate: float
) -> torch.optim.Optimizer:
    """Build AdamW and adapt it for experimental Megatron-FSDP."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, foreach=False)
    if backend == "mfsdpv2":
        fully_shard_optimizer(optimizer)
    return optimizer


def save_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, *, step: int, checkpoint_directory: str
) -> None:
    """Save model, optimizer, and step with PyTorch DCP."""
    if dist.get_rank() == 0:
        logger.info("[rank0] Saving checkpoint step=%d ...", step)
    start = time.time()
    checkpoint_path = Path(checkpoint_directory) / f"step_{step:06d}"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    dcp.save(
        state_dict={"app": AppState(model, optimizer), "step": step},
        checkpoint_id=str(checkpoint_path),
    )
    if dist.get_rank() == 0:
        logger.info("[rank0] Saved checkpoint to %s (%.1fs)", checkpoint_path, time.time() - start)


def load_checkpoint_if_available(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_directory: str
) -> int:
    """Load the latest DCP checkpoint and return the next training step."""
    root = Path(checkpoint_directory)
    checkpoints = sorted(root.glob("step_*")) if root.exists() else []
    if not checkpoints:
        return 0
    checkpoint_path = checkpoints[-1]
    if dist.get_rank() == 0:
        logger.info("[rank0] Loading checkpoint from %s ...", checkpoint_path)
    start = time.time()
    step = torch.zeros((), dtype=torch.int64)
    dcp.load(
        state_dict={"app": AppState(model, optimizer), "step": step},
        checkpoint_id=str(checkpoint_path),
    )
    if dist.get_rank() == 0:
        logger.info("[rank0] Loaded checkpoint (%.1fs)", time.time() - start)
    return int(step.item()) + 1


def _format_bytes(byte_count: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if byte_count < 1024:
            return f"{byte_count:.1f} {unit}"
        byte_count /= 1024
    return f"{byte_count:.1f} TB"


def train(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    start_step: int = 0,
    data: TeacherStudentData | None = None,
) -> None:
    """Train and optionally verify teacher/student convergence."""
    rank = dist.get_rank()
    model.train()
    step = start_step
    start = time.time()
    initial_loss = None
    final_loss = None

    for epoch in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            if data is not None:
                inputs, target = data.sample(
                    batch_size=args.batch_size, sequence_length=args.seq_len, step=step, rank=rank
                )
            else:
                inputs = torch.randn(
                    args.batch_size,
                    args.seq_len,
                    args.model_dim,
                    device=device,
                    dtype=torch.bfloat16,
                )
                target = None

            backward_context = (
                microbatch(model, is_last=True)
                if args.backend == "mfsdpv2" and args.enable_hsdp
                else nullcontext()
            )
            with backward_context:
                output = model(inputs)
                loss = (
                    torch.nn.functional.mse_loss(output.float(), target.float())
                    if target is not None
                    else output.sum() / (args.batch_size * args.seq_len)
                )
                loss.backward()
            optimizer.step()

            global_loss = loss.detach().float()
            dist.all_reduce(global_loss, op=dist.ReduceOp.AVG)
            loss_value = global_loss.item()
            if initial_loss is None:
                initial_loss = loss_value
            final_loss = loss_value

            if step % args.log_interval == 0 and rank == 0:
                elapsed = time.time() - start
                milliseconds_per_step = elapsed / max(step - start_step + 1, 1) * 1000
                logger.info(
                    "[rank0] epoch=%d step=%d loss=%.4e alloc=%s max_reserved=%s "
                    "elapsed=%.1fs (%.0fms/step)",
                    epoch,
                    step,
                    loss_value,
                    _format_bytes(torch.cuda.memory_allocated()),
                    _format_bytes(torch.cuda.max_memory_reserved()),
                    elapsed,
                    milliseconds_per_step,
                )
            if args.ckpt_dir and step > 0 and step % args.ckpt_interval == 0:
                save_checkpoint(model, optimizer, step=step, checkpoint_directory=args.ckpt_dir)
            step += 1

    if args.ckpt_dir:
        save_checkpoint(model, optimizer, step=step, checkpoint_directory=args.ckpt_dir)
    if data is not None and initial_loss is not None and final_loss is not None:
        ratio = final_loss / max(initial_loss, 1.0e-12)
        if rank == 0:
            logger.info(
                "[rank0] convergence: initial_loss=%.4e final_loss=%.4e ratio=%.3f "
                "(threshold=%s)",
                initial_loss,
                final_loss,
                ratio,
                args.convergence_threshold,
            )
        if ratio >= args.convergence_threshold:
            raise AssertionError(
                f"Convergence failed: loss ratio {ratio:.4f} is not below "
                f"{args.convergence_threshold:.4f}."
            )


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "model_dim",
        "n_layers",
        "batch_size",
        "seq_len",
        "epochs",
        "steps_per_epoch",
        "ckpt_interval",
        "log_interval",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least one.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if not 0 < args.convergence_threshold < 1:
        raise ValueError("--convergence-threshold must be between zero and one.")
    if args.backend == "fsdp2" and args.use_symm_mem:
        raise ValueError("--use-symm-mem is supported only by mfsdpv2.")


def _configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main() -> None:
    """Run the toy FSDP example."""
    _configure_logging()
    args = parse_args()
    _validate_args(args)
    device, mesh = initialize_distributed(args.enable_hsdp)
    try:
        if args.record_memory_history:
            torch.cuda.memory._record_memory_history(max_entries=100_000, stacks="all")
        set_seed(args.seed)
        model = ToyModel(dim=args.model_dim, num_layers=args.n_layers).to(
            device=device, dtype=torch.bfloat16
        )
        if args.activation_checkpoint:
            model.enable_activation_checkpointing()
        model = fully_shard_toy_model(
            model, mesh=mesh, backend=args.backend, use_symm_mem=args.use_symm_mem
        )
        optimizer = build_optimizer(model, backend=args.backend, learning_rate=args.lr)
        data = (
            TeacherStudentData(
                dim=args.model_dim, num_layers=args.n_layers, seed=args.seed + 10_000, device=device
            )
            if args.use_real_data
            else None
        )
        start_step = (
            load_checkpoint_if_available(model, optimizer, args.ckpt_dir) if args.ckpt_dir else 0
        )
        train(args, model, optimizer, device=device, start_step=start_step, data=data)
        if args.record_memory_history:
            snapshot_directory = Path(args.record_memory_history)
            snapshot_directory.mkdir(parents=True, exist_ok=True)
            snapshot_path = snapshot_directory / f"memory_snapshot_rank{dist.get_rank()}.pickle"
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            if dist.get_rank() == 0:
                logger.info("[rank0] Memory snapshot dumped to %s", snapshot_path)
    finally:
        if args.record_memory_history:
            torch.cuda.memory._record_memory_history(enabled=None)
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
