import dataclasses

import torch

from .utils import ParamGroupIdx


@dataclasses.dataclass
class Bucket:
    data: torch.Tensor


class BucketAllocator:
    """Interface for allocating and freeing temporary buckets."""

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        """Allocate a bucket for the given param group."""
        raise NotImplementedError

    def free(self, param_group_id: ParamGroupIdx) -> None:
        """Free the bucket associated with the given param group."""
        raise NotImplementedError


class TemporaryBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by param_group_id.

    Used by DataParallelBuffer for unshard (all-gather) and gradient
    reduction (reduce-scatter) operations.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)
            del self.buckets[param_group_id]


class StorageFreeingBucketAllocator(BucketAllocator):
    """Manages temporary flat buffers keyed by param_group_id, and frees the
    underlying storage after use without deleting the bucket entry, so the
    same tensor object can be reused on the next allocation.
    """

    def __init__(self):
        super().__init__()
        self.buckets = {}

    def allocate(
        self, param_group_id: ParamGroupIdx, size: int, dtype: torch.dtype, device: torch.device
    ) -> Bucket:
        if param_group_id not in self.buckets:
            self.buckets[param_group_id] = Bucket(
                data=torch.empty(size, dtype=dtype, device=device)
            )
            return self.buckets[param_group_id]
        _alloc_storage(self.buckets[param_group_id].data, torch.Size([size]))
        return self.buckets[param_group_id]

    def free(self, param_group_id: ParamGroupIdx) -> None:
        if param_group_id in self.buckets:
            _free_storage(self.buckets[param_group_id].data)


def _free_storage(tensor: torch.Tensor) -> None:
    """Free the underlying storage of ``tensor`` by resizing it to 0."""
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                assert tensor.storage_offset() == 0, (
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}"
                )
                tensor._typed_storage()._resize_(0)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """Re-allocate storage for ``tensor`` to the given ``size``.

    Requires that the tensor's storage has been freed (resized to 0)
    before calling.  The caller must ensure ``size`` matches the tensor's
    existing shape.
    """
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                assert tensor_storage_size == 0, (
                    "Tensor storage should have been resized to 0 but got "
                    f"{tensor_storage_size} (shape={tensor.shape})"
                )
                tensor._typed_storage()._resize_(size.numel())
