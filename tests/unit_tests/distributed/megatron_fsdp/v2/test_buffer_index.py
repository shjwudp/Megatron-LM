# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from pathlib import Path

import pytest
import torch
from torch.distributed.tensor import DeviceMesh

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import BufferIndex
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx


@pytest.fixture(scope="session", autouse=True)
def dist_env():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(torch.device(f"cuda:{rank % torch.cuda.device_count()}"))
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _build_hsdp_mesh():
    world_size = torch.distributed.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("BufferIndex HSDP layout coverage requires an even world size >= 4")
    mesh = torch.arange(world_size, dtype=torch.int).reshape(2, world_size // 2)
    return DeviceMesh("cuda", mesh, mesh_dim_names=("dp_outer", "dp"))


def _intersect(left, right):
    start = max(left[0], right[0])
    end = min(left[1], right[1])
    return None if start >= end else (start, end)


def _meta_range(meta):
    return (meta.global_data_index, meta.global_data_index + meta.size)


@pytest.mark.parametrize(
    ("inner_sharded", "outer_sharded"),
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("shard_level", ["full", "inner", "outer"])
def test_buffer_index_item_ranges(inner_sharded, outer_sharded, shard_level):
    mesh = _build_hsdp_mesh()
    index = BufferIndex(
        param_shapes=[torch.Size([64])],
        mesh=mesh,
        inner_sharded=inner_sharded,
        outer_sharded=outer_sharded,
        param_group_id=ParamGroupIdx(0, 0),
        chunk_size_factor=1,
    )

    item_range = index._get_item_global_range(0)
    if shard_level == "full":
        requested_range = item_range
    elif shard_level == "inner":
        requested_range = _intersect(item_range, _meta_range(index.shard_meta))
    else:
        requested_range = _intersect(item_range, _meta_range(index.outer_shard_meta))

    if requested_range is None:
        expected_self = (0, 0)
    else:
        expected_self = (
            requested_range[0] - item_range[0],
            requested_range[1] - item_range[0],
        )
    assert index._get_item_self_range(0, shard_level=shard_level) == expected_self

    if outer_sharded:
        storage_meta = index.outer_shard_meta
    elif inner_sharded:
        storage_meta = index.shard_meta
    else:
        storage_meta = None

    local_range = requested_range
    if storage_meta is not None and local_range is not None:
        local_range = _intersect(local_range, _meta_range(storage_meta))

    if local_range is None:
        expected_local = (0, 0)
    elif storage_meta is None:
        expected_local = local_range
    else:
        expected_local = (
            storage_meta.local_data_index + local_range[0] - storage_meta.global_data_index,
            storage_meta.local_data_index + local_range[1] - storage_meta.global_data_index,
        )
    assert index._get_item_local_range(0, shard_level=shard_level) == expected_local

    if outer_sharded and shard_level == "outer":
        assert expected_local == (0, index.outer_shard_meta.size)
