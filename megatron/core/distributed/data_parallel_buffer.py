import math
from collections import namedtuple
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

TensorItemIndex = namedtuple(
    'TensorItemIndex', ['global_data_index', 'size', 'item_id', 'bucket_id']
)
BucketIndex = namedtuple('BucketIndex', ['global_data_index', 'size', 'items'])
ShardBucketIndex = namedtuple(
    'ShardBucketIndex', ['global_data_index', 'local_data_index', 'bucket_data_index', 'size']
)


@dataclass
class Bucket:
    id: int
    is_shard: bool
    data: torch.Tensor
    items: List[int]
    requires_grad_items: List[TensorItemIndex]


def bucket_size_pad(x: int, x_based: int) -> int:
    return int(math.ceil(x / x_based)) * x_based


def build_data_parallel_buffer_index(
    elements: List[torch.Size],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    guide_bucket_size: int,
) -> Tuple[int, List[tuple], List[tuple], List[tuple]]:
    """
    Assuming that all input tensor elements are consecutively compose a global 
    buffer, give the index range of every tensor,  every bucket and every in 
    bucket local buffer.

    Args:
        elements (List[torch.Size]): List of input tensor.
        data_parallel_world_size (int): The number of data parallel world size.
        guide_bucket_size (int, optional): The guide bucket size. Defaults to 40_000_000.
    
    Returns:
        int: The total size of the global buffer.
        List[tuple]: The index range of every tensor.
        List[tuple]: The index range of every bucket.
        List[tuple]: The index range of every in bucket local buffer.
    """
    item_index_map = []
    bucket_index_map = []
    bucket_id = 0
    bucket = []
    data_index = 0
    for item in elements:
        item_id = len(item_index_map)
        bucket_id = len(bucket_index_map)
        bucket.append(item)
        bucket_size = sum([it.numel() for it in bucket])
        item_index_map.append(
            TensorItemIndex(
                data_index + bucket_size - item.numel(),
                item.numel(),
                item_id=item_id,
                bucket_id=bucket_id,
            )
        )
        if bucket_size >= guide_bucket_size:
            bucket_size = bucket_size_pad(bucket_size, x_based=data_parallel_world_size)
            bucket_index_map.append(
                BucketIndex(
                    data_index,
                    bucket_size,
                    items=list(filter(lambda x: x.bucket_id == bucket_id, item_index_map)),
                )
            )
            data_index += bucket_size
            bucket.clear()

    if len(bucket) > 0:
        bucket_size = bucket_size_pad(bucket_size, x_based=data_parallel_world_size)
        bucket_index_map.append(
            BucketIndex(
                data_index,
                bucket_size,
                items=list(filter(lambda x: x.bucket_id == bucket_id, item_index_map)),
            )
        )
        data_index += bucket_size

    shard_bucket_index_map = []
    local_data_index = 0
    for bucket_index in bucket_index_map:
        shard_size = bucket_index.size // data_parallel_world_size
        bucket_data_index = shard_size * data_parallel_rank
        global_data_index = bucket_index.global_data_index + bucket_data_index
        shard_bucket_index_map.append(
            ShardBucketIndex(global_data_index, local_data_index, bucket_data_index, shard_size),
        )
        local_data_index += shard_size

    return data_index, item_index_map, bucket_index_map, shard_bucket_index_map


class DataParallelBuffer:
    def __init__(
        self,
        data_parallel_group: torch.distributed.ProcessGroup,
        parameters: List[torch.nn.Parameter],
        dtype: torch.dtype,
        device: torch.device = torch.cuda.current_device(),
        guide_bucket_size: int = 40_000_000,
    ):
        self.data_parallel_group = data_parallel_group
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.data_parallel_world_size = torch.distributed.get_world_size(group=data_parallel_group)
        self.dtype = dtype
        self.device = device
        self.parameters = parameters
        (
            self.global_buffer_size,
            self.item_index_map,
            self.bucket_index_map,
            self.shard_bucket_index_map,
        ) = build_data_parallel_buffer_index(
            [p.shape for p in parameters],
            self.data_parallel_rank,
            self.data_parallel_world_size,
            guide_bucket_size,
        )
        local_buffer_size = sum([x.size for x in self.shard_bucket_index_map])
        self.data = torch.zeros(local_buffer_size, dtype=dtype, device=device)
        self.buckets = {}
        self.reduce_scatter_count = 0

    def get_bucket_local_sharding(self, bucket_id: int) -> torch.Tensor:
        """Get the local sharding of a bucket by bucket id."""
        index = self.shard_bucket_index_map[bucket_id]
        return self.data[index.local_data_index : index.local_data_index + index.size]

    @torch.no_grad()
    def put_into_bucket(self, item_id: int, data: torch.Tensor) -> None:
        item_index = self.item_index_map[item_id]
        bucket_id = item_index.bucket_id
        bucket_index = self.bucket_index_map[bucket_id]
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = self.allocate_bucket(bucket_id)
        bucket = self.buckets[bucket_id]

        offset = item_index.global_data_index - bucket_index.global_data_index
        bucket.data[offset : offset + item_index.size] = data.flatten()
        bucket.items.append(item_id)

        if len(bucket.items) == len(bucket.requires_grad_items):
            shard_bucket_index = self.shard_bucket_index_map[bucket_id]
            offset = shard_bucket_index.bucket_data_index
            shard_size = shard_bucket_index.size
            shard = bucket.data[offset : offset + shard_size]

            bucket.data *= 1.0 / self.data_parallel_world_size
            torch.distributed.reduce_scatter_tensor(
                output=shard, input=bucket.data, op=torch.distributed.ReduceOp.SUM,
                group=self.data_parallel_group,
            )

            # gradient accumulation on local buffer
            local_buffer = self.get_bucket_local_sharding(bucket_id)
            local_buffer += shard

            del self.buckets[bucket_id]
            self.reduce_scatter_count += 1

    def allocate_bucket(self, bucket_id: int, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Allocate a full size bucket by bucket id."""
        if dtype is None:
            dtype = self.dtype

        bucket_index = self.bucket_index_map[bucket_id]
        requires_grad_items = []
        for item_index in bucket_index.items:
            if self.parameters[item_index.item_id].requires_grad:
                requires_grad_items.append(item_index)
        return Bucket(
            bucket_id,
            False,
            torch.zeros(bucket_index.size, dtype=dtype, device=self.device),
            items=[],
            requires_grad_items=requires_grad_items,
        )

    def _get_item_slice(self, item_id: int) -> Tuple[int, int]:
        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index_map[item_index.bucket_id]

        item_global_start = item_index.global_data_index
        item_global_end = item_index.global_data_index + item_index.size
        shard_bucket_start = shard_bucket_index.global_data_index
        shard_bucket_end = shard_bucket_index.global_data_index + shard_bucket_index.size

        if item_global_start > shard_bucket_end or item_global_end < shard_bucket_start:
            return (-1, -1)

        start = max(item_global_start, shard_bucket_start) - item_global_start
        end = min(item_global_end, shard_bucket_end) - item_global_start

        return (start, end)

    def _get_item_local_index(self, item_id: int) -> Tuple[int, int]:
        item_index = self.item_index_map[item_id]
        shard_bucket_index = self.shard_bucket_index_map[item_index.bucket_id]
        slice_start, slice_end = self._get_item_slice(item_id)

        if slice_start == -1 or slice_end == -1:
            return (-1, -1)

        index_diff = item_index.global_data_index - shard_bucket_index.global_data_index + shard_bucket_index.local_data_index

        return (slice_start + index_diff, slice_end + index_diff)

    def get_item(self, item_id: int) -> torch.Tensor:
        start, end = self._get_item_local_index(item_id)
        return self.data[start:end]

    @torch.no_grad()
    def set_item(self, item_id: int, item_data: torch.Tensor) -> None:
        slice_start, slice_end = self._get_item_slice(item_id)
        local_index_start, local_index_end = self._get_item_local_index(item_id)

        self.data[local_index_start:local_index_end] = item_data.flatten()[slice_start:slice_end]
