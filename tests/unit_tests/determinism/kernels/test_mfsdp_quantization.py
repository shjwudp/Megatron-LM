"""Bit-exact replay of MFSDP's padded MXFP8 quantization outputs."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("transformer_engine.pytorch")
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer import (
    QuantizedDBuffer,
)

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="MXFP8 quantization requires Blackwell or newer",
    ),
]


@pytest.mark.parametrize("shape", [(32, 4096), (128, 4096)])
def test_mfsdp_quantization_replays(shape):
    Utils.initialize_model_parallel()
    try:
        seeded()
        device = torch.device("cuda", torch.cuda.current_device())
        mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),))
        quantized = QuantizedDBuffer.empty(mesh, [Replicate()], [shape], device)
        master = DBuffer(
            mesh, [Replicate()], quantized.rowwise_data.layout, torch.float32, device
        )

        def quantize(source):
            master.get_tensor_view(0).copy_(source)
            quantized.quantize_(master)
            view = quantized.get_tensor_view(0)
            return tuple(
                getattr(view, attribute)
                for attribute in (
                    "_rowwise_data",
                    "_columnwise_data",
                    "_rowwise_scale_inv",
                    "_columnwise_scale_inv",
                )
            )

        source = torch.randn(shape, device=device, dtype=torch.float32)
        assert_replays_bit_exact(
            quantize,
            (source,),
            backward=False,
            replays=3,
            contention=True,
            what="MFSDP MXFP8 quantization",
        )
    finally:
        Utils.destroy_model_parallel()
