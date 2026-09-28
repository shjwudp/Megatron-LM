# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MXFP8 payload detachment: ``clear_payloads`` empties both raw payload directions."""

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    clear_payloads,
)


def test_clear_payloads_detaches_both_payloads():
    """Detaching a quantized tensor drops its row-wise and column-wise payloads."""
    tensor = torch.zeros(4, 4)
    tensor._rowwise_data = torch.zeros(4, 4)
    tensor._columnwise_data = torch.zeros(4, 4)
    clear_payloads(tensor)
    assert tensor._rowwise_data is None
    assert tensor._columnwise_data is None
