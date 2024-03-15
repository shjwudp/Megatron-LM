import math
from typing import List
from collections import namedtuple

import torch


TensorItemIndex = namedtuple('TensorItemIndex', ['global_data_index', 'size', 'item_id', 'bucket_id'])
BucketIndex = namedtuple('BucketIndex', ['global_data_index', 'size', 'items'])
ShardBucketIndex = namedtuple('ShardBucketIndex', ['global_data_index', 'local_data_index', 'size'])
Bucket = namedtuple('Bucket', ['id', 'is_shard', 'data', 'items', 'requires_grad_items'])


def bucket_size_pad(x: int, x_based: int) -> int:
    return int(math.ceil(x / x_based)) * x_based


def build_data_parallel_buffer_index(
    elements: List[torch.Tensor.shape],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    guide_bucket_size: int,
) -> tuple[int, List[tuple], List[tuple], List[tuple]]:
    """
    Assuming that all input tensor elements are consecutively compose a global 
    buffer, give the index range of every tensor,  every bucket and every in 
    bucket local buffer.

    Args:
        elements (List[torch.Tensor]): List of input tensor.
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
        item_index_map.append(TensorItemIndex(
            data_index + bucket_size - item.numel(),
            bucket_size,
            item_id=item_id,
            bucket_id=bucket_id,
        ))
        if bucket_size >= guide_bucket_size:
            bucket_size = bucket_size_pad(bucket_size, x_based=data_parallel_world_size)
            bucket_index_map.append(BucketIndex(
                data_index,
                bucket_size,
                items=list(filter(lambda x: x.bucket_id == bucket_id, item_index_map)),
            ))
            data_index += bucket_size
            bucket.clear()

    if len(bucket) > 0:
        bucket_size = bucket_size_pad(bucket_size, x_based=data_parallel_world_size)
        bucket_index_map.append(BucketIndex(
            data_index,
            bucket_size,
            items=list(filter(lambda x: x.bucket_id == bucket_id, item_index_map)),
        ))
        data_index += bucket_size

    shard_bucket_index_map = []
    local_data_index = 0
    for bucket_start_idx, bucket_end_idx in bucket_index_map:
        bucket_size = bucket_end_idx - bucket_start_idx
        shard_size = bucket_size // data_parallel_world_size
        
        global_data_index = bucket_start_idx + shard_size * data_parallel_rank
        shard_bucket_index_map.append(
            ShardBucketIndex(global_data_index, local_data_index, shard_size),
        )

    return data_index, item_index_map, bucket_index_map, shard_bucket_index_map


class DataParallelBuffer:

    def __init__(
        self,
        data_parallel_rank: int,
        data_parallel_world_size: int,
        dtype: torch.dtype,
        parameters: List[torch.Tensor],
        guide_bucket_size: int = 40_000_000,
    ):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.dtype = dtype
        self.parameters = parameters
        self.data_index, self.item_index_map, self.bucket_index_map, self.shard_bucket_index_map = \
            build_data_parallel_buffer_index(
                [p.shape for p in parameters],
                data_parallel_rank,
                data_parallel_world_size,
                guide_bucket_size,
            )
        local_buffer_size = sum([x.size for x in self.shard_bucket_index_map])
        self.buffer = torch.zeros(local_buffer_size, dtype=dtype)
        self.buckets = {}
        self.reduce_scatter_count = 0

    def get_bucket_local_sharding(self, bucket_id: int) -> torch.Tensor:
        """Get the local sharding of a bucket by bucket id."""
        index = self.bucket_index_map[bucket_id]
        return self.buffer[index.global_data_index: index.global_data_index + index.size]

    def put_into_bucket(self, item_id: int, data: torch.Tensor) -> None:
        item_index = self.item_index_map[item_id]
        bucket_id = item_index.bucket_id
        bucket_index = self.bucket_index_map[bucket_id]
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = self.allocate_bucket(bucket_id)
        bucket = self.buckets[bucket_id]

        offset = item_index.global_data_index - bucket_index.global_data_index
        bucket.data[offset: offset + item_index.size] = data
        bucket.items.append(item_id)

        if len(self.items) == len(bucket.requires_grad_items):
            bucket.data *= 1.0 / self.data_parallel_world_size
            torch.distributed.reduce_scatter_tensor(
                output=self.get_bucket_local_sharding(bucket_id),
                input=bucket.data,
                op=torch.distributed.ReduceOp.SUM,
            )
            del self.buckets[bucket_id]
            self.reduce_scatter_count += 1

    def allocate_bucket(self, bucket_id: int) -> torch.Tensor:
        """Allocate a full size bucket by bucket id."""
        bucket_index = self.bucket_index_map[bucket_id]        
        requires_grad_items = []
        for item_index in bucket_index.items:
            if self.parameters[item_index.item_id].requires_grad:
                requires_grad_items.append(item_index)
        return Bucket(
            bucket_id,
            False,
            torch.zeros(bucket_index.size, dtype=self.dtype),
            items=[],
            requires_grad_items=requires_grad_items,
        )
