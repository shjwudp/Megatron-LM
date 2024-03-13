import math
from typing import List
from collections import namedtuple

import torch

from megatron.core import parallel_state


TensorItemIndex = namedtuple('TensorItemIndex', ['global_data_index', 'size', 'bucket_id'])
BucketIndex = namedtuple('BucketIndex', ['global_data_index', 'size'])
ShardBucketIndex = namedtuple('ShardBucketIndex', ['global_data_index', 'local_data_index', 'size'])


def build_data_parallel_buffer_index(
    elements: List[torch.Tensor.shape],
    data_parallel_rank: int,
    data_parallel_world_size: int,
    guide_bucket_size: int = 40_000_000,
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
        bucket_id = len(bucket_index_map)
        bucket.append(item)
        bucket_size = sum([it.numel() for it in bucket])
        item_index_map.append((
            data_index + bucket_size - item.numel(),
            bucket_size,
            bucket_id,
        ))
        if bucket_size >= guide_bucket_size:
            bucket_size = bucket_size_pad(bucket_size, x_based=data_parallel_world_size)
            bucket_index_map.append(TensorItemIndex(
                data_index,
                bucket_size,
                bucket_id,
            ))
            data_index += bucket_size
            bucket.clear()

    if len(bucket) > 0:
        bucket_size = bucket_size_pad(bucket_size, x_based=data_parallel_world_size)
        bucket_index_map.append(BucketIndex(
            data_index,
            bucket_size,
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
        elements: List[torch.Tensor.shape],
    ):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.dtype = dtype
        self.data_index, self.item_index_map, self.bucket_index_map, self.shard_bucket_index_map = \
            build_data_parallel_buffer_index(
                elements,
                data_parallel_rank,
                data_parallel_world_size,
            )
        local_buffer_size = sum([x.size for x in self.shard_bucket_index_map])
        self.buffer = torch.zeros(local_buffer_size, dtype=dtype)
