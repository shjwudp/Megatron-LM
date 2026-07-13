# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Shared utilities for toy_model examples."""

import atexit
import os
from pathlib import Path

import torch
import torch.distributed as dist


def fmt_bytes(n: int) -> str:
    for power, suffix in [(4, "TB"), (3, "GB"), (2, "MB"), (1, "KB"), (0, "B")]:
        unit = 1024**power
        if n >= unit:
            return f"{n / unit:.2f} {suffix}"
    return f"{n} B"


def mem_log(tag="", rank=None):
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    alloc = torch.cuda.memory_allocated()
    max_alloc = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_reserved = torch.cuda.max_memory_reserved()
    prefix = f"[rank{rank}] {tag}" if tag else f"[rank{rank}]"
    print(f"{prefix} alloc={fmt_bytes(alloc)} max_alloc={fmt_bytes(max_alloc)} "
          f"reserved={fmt_bytes(reserved)} max_reserved={fmt_bytes(max_reserved)}")


class MemoryHistoryManager:
    """Records CUDA memory history and dumps snapshots on OOM and/or exit."""

    def __init__(self, out_dir: str, oom_only: bool = False):
        self._out_dir = out_dir
        self._oom_only = oom_only
        self._dumped = False

    def start(self):
        Path(self._out_dir).mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=200000, stacks="all")
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[rank0] Memory history recording enabled, "
                  f"dump dir={self._out_dir} oom_only={self._oom_only}")

    def dump(self, tag: str = ""):
        if self._dumped:
            return
        self._dumped = True
        rank = dist.get_rank() if dist.is_initialized() else 0
        suffix = f"_{tag}" if tag else ""
        path = os.path.join(self._out_dir, f"memory_snapshot_rank{rank}{suffix}.pickle")
        try:
            torch.cuda.memory._dump_snapshot(path)
            if rank == 0:
                print(f"[rank0] Memory snapshot dumped: {path}")
        except Exception as e:
            if rank == 0:
                print(f"[rank0] Memory snapshot dump failed: {e}")

    def stop(self):
        torch.cuda.memory._record_memory_history(enabled=None)

    def dump_on_normal_exit(self):
        if self._oom_only or self._dumped:
            return
        self.dump()

    def dump_on_oom(self):
        self.dump(tag="OOM")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> torch.distributed.device_mesh.DeviceMesh:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    from torch.distributed.device_mesh import init_device_mesh
    return init_device_mesh("cuda", mesh_shape=(dist.get_world_size(),))
