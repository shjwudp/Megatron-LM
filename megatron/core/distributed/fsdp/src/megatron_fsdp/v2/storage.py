# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Low-level tensor-storage helpers shared by buffers and allocators."""

import torch


def _is_torchdynamo_compiling() -> bool:
    """Check whether torchdynamo is compiling, safely across PyTorch versions."""
    try:
        return torch.distributed._functional_collectives.is_torchdynamo_compiling()
    except (AttributeError, RuntimeError):
        return False


def free_storage(tensor: torch.Tensor) -> None:
    """Free the underlying storage of ``tensor`` by resizing it to zero."""
    with torch.no_grad():
        if not _is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                assert tensor.storage_offset() == 0, (
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}"
                )
                tensor._typed_storage()._resize_(0)


def alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """Reallocate previously freed ``tensor`` storage to ``size``."""
    with torch.no_grad():
        if not _is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                assert tensor_storage_size == 0, (
                    "Tensor storage should have been resized to 0 but got "
                    f"{tensor_storage_size} (shape={tensor.shape})"
                )
                tensor._typed_storage()._resize_(size.numel())
