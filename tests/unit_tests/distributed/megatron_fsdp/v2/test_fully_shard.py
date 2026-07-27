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

"""Unit tests for the Megatron-FSDP v2 ``fully_shard`` API, ``FSDPModule``,
and checkpoint (``get_state_dict``).

Covers:
- Basic fully_shard: class mutation, hooks, reshard on init
- Multi-layer LLM-style nesting (embedding → transformer layers → lm_head)
- Multimodal-style: separate encoders with partial freezing
- Partially frozen training (requires_grad=False on some params)
- Nested FSDP (expert-in-layer pattern)
- Mixed precision policies (fp32 main params, fp32 grad reduce)
- ignored_params
- enable_unshard_prefetch / enable_async_reduce_grad feature flags
- Forward/backward lifecycle correctness
- get_state_dict / preprocess_state_dict_for_uneven_dtensor
- Double-shard safety (reject re-wrap)

Run with:
    torchrun --nproc_per_node=2 -m pytest \\
        tests/unit_tests/distributed/megatron_fsdp/v2/test_fully_shard.py -v

Single-GPU tests:
    pytest tests/unit_tests/distributed/megatron_fsdp/v2/test_fully_shard.py -v \\
        -k "test_double_shard_rejected or test_no_params_module"
"""

import shutil
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions
from torch.distributed.checkpoint.state_dict import get_state_dict as torch_get_state_dict
from torch.distributed.checkpoint.state_dict import set_state_dict as torch_set_state_dict
from torch.distributed.tensor import DeviceMesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.uneven_dtensor import (
    get_state_dict,
    preprocess_state_dict_for_uneven_dtensor,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.allocator import TracePoolAllocator
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.buffer_index import Placement
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fsdp_module import FSDPModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.fully_shard import fully_shard
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.hooks import (
    mfsdp_forward_pre_hook,
    mfsdp_post_backward_final_callback,
    mfsdp_post_backward_hook,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.mixed_precision import MixedPrecisionPolicy
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.param_group import (
    GradientPhase,
    ParameterGroup,
)

SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"

# ------------------------------------------------------------------ #
#  Distributed environment (NCCL session-scoped)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="session", autouse=True)
def dist_env():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _rank():
    return torch.distributed.get_rank()


def _world_size():
    return torch.distributed.get_world_size()


def _device():
    return torch.device(f"cuda:{_rank() % torch.cuda.device_count()}")


def _build_hsdp_mesh():
    world_size = _world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("HSDP checkpoint coverage requires an even world size >= 4")

    mesh = torch.arange(world_size, dtype=torch.int).reshape(2, world_size // 2)
    return DeviceMesh(_device().type, mesh, mesh_dim_names=("dp_outer", "dp"))


# ------------------------------------------------------------------ #
#  Mock models for different application scenarios
# ------------------------------------------------------------------ #


class SimpleMLP(nn.Module):
    """Single linear layer with optional bias."""

    def __init__(self, hidden=64, bias=True):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden, bias=bias)

    def forward(self, x):
        return self.fc(x)


class MixedDtypeBuffers(nn.Module):
    """Module whose FSDP unit owns communication buffers with different dtypes."""

    def __init__(self, hidden=64):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(hidden, hidden))
        nn.init.normal_(self.weight)
        self.bfloat16_weight = nn.Parameter(
            torch.arange(hidden * hidden, dtype=torch.bfloat16).reshape(hidden, hidden),
            requires_grad=False,
        )

    def forward(self, x):
        return x @ self.weight


class TinyLLM(nn.Module):
    """Simulates an LLM: embedding → block of layers → lm_head.

    Structure::
        embedding (nn.Embedding) → layers (nn.ModuleList of SimpleMLP) → lm_head (nn.Linear)
    """

    def __init__(self, vocab=128, hidden=64, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([SimpleMLP(hidden) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden, vocab)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.norm(h))


class MultimodalModel(nn.Module):
    """Simulates a multimodal model with separate vision/text encoders.

    Structure::
        vision_encoder (nn.Linear) — may be frozen
        text_encoder (nn.Linear) — trainable
        fusion (nn.Linear) — trainable
    """

    def __init__(self, hidden=64):
        super().__init__()
        self.vision_encoder = nn.Linear(hidden, hidden)
        self.text_encoder = nn.Linear(hidden, hidden)
        self.fusion = nn.Linear(hidden * 2, hidden)

    def forward(self, img, txt):
        v = self.vision_encoder(img)
        t = self.text_encoder(txt)
        return self.fusion(torch.cat([v, t], dim=-1))


class ExpertBlock(nn.Module):
    """Simulates an MoE expert: two linear layers."""

    def __init__(self, hidden=64, ffn_hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn_hidden)
        self.fc2 = nn.Linear(ffn_hidden, hidden)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MOETransformerLayer(nn.Module):
    """Simulates a transformer layer with MoE: attention → MoE experts.

    Structure::
        attn (nn.Linear) → experts (ExpertBlock) → norm (nn.LayerNorm)
    """

    def __init__(self, hidden=64, ffn_hidden=128):
        super().__init__()
        self.attn = nn.Linear(hidden, hidden)
        self.experts = ExpertBlock(hidden, ffn_hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        h = self.attn(x)
        h = self.experts(h)
        return self.norm(h + x)


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def _forward_backward(model, x):
    """Run forward + backward and return loss."""
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss.item()


def _assert_dtensor_params(module):
    """Assert all parameters in the module (and any nested FSDPModules) are DTensors."""
    from torch.distributed.tensor import DTensor

    for name, p in module.named_parameters():
        assert isinstance(p, DTensor), (
            f"Parameter '{name}' should be a DTensor after fully_shard, " f"got {type(p).__name__}"
        )


def _assert_original_params_unchanged(module, originals):
    """After fully_shard, the original (pre-fully_shard) param OBJECTS should
    still be the same Python objects (identity check), but their .data may have
    been freed (empty tensor)."""
    for name, p in module.named_parameters():
        assert (
            p is originals[name]
        ), f"Original param object for '{name}' was replaced; expected identity match."


def _count_fsdp_modules(module):
    """Return number of FSDPModule instances in the module tree."""
    return sum(1 for m in module.modules() if isinstance(m, FSDPModule))


# ------------------------------------------------------------------ #
#  1. Basic fully_shard — class mutation, hooks, reshard on init
# ------------------------------------------------------------------ #


class TestFullyShardBasic:
    def test_module_class_becomes_fsdp(self):
        """fully_shard should dynamically convert the module class to a FSDPModule mixin."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        original_cls = model.__class__
        wrapped = fully_shard(model)
        assert wrapped is model  # returns same object
        assert isinstance(wrapped, FSDPModule)
        assert FSDPModule in type(wrapped).__mro__
        assert original_cls in type(wrapped).__mro__

    def test_params_are_dtensor_after_reshard(self):
        """After fully_shard, module.reshard() is called, so params must be DTensors."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        _assert_dtensor_params(model)

    def test_forward_without_errors(self):
        """A simple forward pass after fully_shard should succeed."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        x = torch.randn(2, 64, device=_device())
        out = model(x)
        assert out.shape == (2, 64)

    def test_forward_backward_no_nan(self):
        """Forward + backward should produce finite loss and gradients."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        x = torch.randn(2, 64, device=_device())
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
        assert not torch.isinf(torch.tensor(loss)), "Loss is Inf"

    @pytest.mark.parametrize(
        "sharding_strategy", ["no_shard", "optim", "optim_grads", "optim_grads_params"]
    )
    @pytest.mark.parametrize(
        ("model_dtype", "main_dtype"),
        [
            pytest.param(torch.float32, None, id="fp32"),
            pytest.param(torch.bfloat16, torch.float32, id="bf16-fp32-main"),
            pytest.param(torch.float16, torch.float32, id="fp16-fp32-main"),
        ],
    )
    def test_parameter_group_eager_1d(self, sharding_strategy, model_dtype, main_dtype):
        """The experimental eager path runs the complete 1D FSDP lifecycle."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device(), dtype=model_dtype)
        fully_shard(
            model,
            sharding_strategy=sharding_strategy,
            mp_policy=MixedPrecisionPolicy(
                main_params_dtype=main_dtype, main_grads_dtype=main_dtype
            ),
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
        )

        assert model._fsdp_param_groups
        assert all(
            isinstance(param_group, ParameterGroup) and param_group.mesh.ndim == 1
            for param_group in model._fsdp_param_groups
        )

        model.set_is_last_backward(True)
        loss = _forward_backward(model, torch.randn(2, 16, device=_device(), dtype=model_dtype))
        assert torch.isfinite(torch.tensor(loss))
        model.finish_grad_sync()

        for param_group in model._fsdp_param_groups:
            assert param_group.state.grad_phase is GradientPhase.READY
            for optimizer_param in param_group.optimizer_params:
                if optimizer_param.requires_grad:
                    assert optimizer_param.grad is not None

        model.zero_grad(set_to_none=True)
        for param_group in model._fsdp_param_groups:
            assert param_group.state.grad_phase is GradientPhase.EMPTY
            assert param_group.grad_buffer.data is None

    @pytest.mark.parametrize("sharding_strategy", ["no_shard", "optim_grads_params"])
    def test_parameter_group_per_module_cuda_graph(self, sharding_strategy):
        """Per-module capture preserves placement-first gradient accumulation."""
        model = fully_shard(
            SimpleMLP(4).to(_device()),
            sharding_strategy=sharding_strategy,
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
            enable_cuda_graph=True,
        )

        for index, value in enumerate((2.0, 3.0, 4.0)):
            if index == 2:
                model.set_is_last_backward(True)
            sample = torch.full((2, 4), value, device=_device(), requires_grad=True)
            model(sample).sum().backward()
            if index == 1:
                assert model._fsdp_cg_installed

        model.finish_grad_sync()
        for param_names, param_group in model._named_param_groups:
            assert param_group.state.grad_phase is GradientPhase.READY
            for name, optimizer_grad in zip(param_names, param_group.optimizer_grads):
                if optimizer_grad is None:
                    continue
                local_expected = 18.0 if name.endswith("weight") else 6.0
                expected = local_expected * _world_size()
                torch.testing.assert_close(
                    optimizer_grad.to_local(),
                    torch.full_like(optimizer_grad.to_local(), expected),
                )

        model.zero_grad()
        for param_group in model._fsdp_param_groups:
            assert param_group.state.grad_phase is GradientPhase.EMPTY

    @pytest.mark.parametrize(
        "sharding_strategy,outer_dp_sharding_strategy",
        [
            ("no_shard", "no_shard"),
            ("optim", "no_shard"),
            ("optim_grads", "no_shard"),
            ("optim_grads_params", "no_shard"),
            ("optim_grads_params", "optim"),
        ],
    )
    def test_parameter_group_eager_hsdp(
        self, sharding_strategy, outer_dp_sharding_strategy
    ):
        """The eager HSDP path preserves the caller's 2D mesh and final layout."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        fully_shard(
            model,
            mesh=_build_hsdp_mesh(),
            sharding_strategy=sharding_strategy,
            outer_dp_sharding_strategy=outer_dp_sharding_strategy,
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
        )

        model.set_is_last_backward(True)
        loss = _forward_backward(model, torch.randn(2, 16, device=_device()))
        assert torch.isfinite(torch.tensor(loss))
        model.finish_grad_sync()

        expected_outer = (
            Placement.SHARD if outer_dp_sharding_strategy == "optim" else Placement.REPLICATE
        )
        expected_inner = (
            Placement.REPLICATE
            if sharding_strategy == "no_shard"
            else Placement.SHARD
        )
        for param_group in model._fsdp_param_groups:
            assert isinstance(param_group, ParameterGroup)
            assert param_group.mesh.ndim == 2
            assert param_group.layout.main_weight == (expected_outer, expected_inner)
            assert param_group.state.grad_phase is GradientPhase.READY

    def test_parameter_group_hsdp_axis_streams(self):
        """HSDP accepts independent outer/inner all-gather and reduction streams."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        outer_ag_stream = torch.cuda.Stream()
        inner_ag_stream = torch.cuda.Stream()
        outer_rs_stream = torch.cuda.Stream()
        inner_rs_stream = torch.cuda.Stream()
        fully_shard(
            model,
            mesh=_build_hsdp_mesh(),
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy="optim",
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=True,
            all_gather_streams=(outer_ag_stream, inner_ag_stream),
            reduce_scatter_streams=(outer_rs_stream, inner_rs_stream),
        )

        ctx = model._fsdp_root_context
        assert ctx.ag_streams == (outer_ag_stream, inner_ag_stream)
        assert ctx.rs_streams == (outer_rs_stream, inner_rs_stream)

        model.set_is_last_backward(True)
        loss = _forward_backward(model, torch.randn(2, 16, device=_device()))
        model.finish_grad_sync()
        assert torch.isfinite(torch.tensor(loss))
        assert all(
            param_group.state.grad_phase is GradientPhase.READY
            for param_group in model._fsdp_param_groups
        )

    @pytest.mark.parametrize("depth", [None, 0, 1, 2])
    def test_hsdp_outer_weight_prefetch_window(self, depth):
        """The first unshard bootstraps and advances the outer prefetch window."""

        class LayerStack(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([SimpleMLP(16) for _ in range(4)])

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        mesh = _build_hsdp_mesh()
        model = LayerStack().to(_device())
        shard_kwargs = {
            "mesh": mesh,
            "sharding_strategy": "optim_grads_params",
            "outer_dp_sharding_strategy": "optim",
            "enable_unshard_prefetch": True,
            "enable_async_reduce_grad": False,
        }
        for layer in model.layers:
            fully_shard(layer, **shard_kwargs)
        root_kwargs = dict(shard_kwargs)
        if depth is not None:
            root_kwargs["outer_dp_all_gather_prefetch_depth"] = depth
        fully_shard(model, **root_kwargs)
        effective_depth = 1 if depth is None else depth

        # Model an optimizer update: only each [S, S] main-weight view is current.
        model._copy_main_weights_to_model_weights()

        # Layer 0 performs its own outer then inner AG before refilling exactly
        # the configured number of future outer stages.
        model.layers[0].unshard(async_op=True)
        assert not any(
            param_group.state.pending_weights
            for param_group in model.layers[0]._fsdp_param_groups
        )
        pending = [
            any(param_group.state.pending_weights for param_group in layer._fsdp_param_groups)
            for layer in model.layers
        ]
        assert pending == [
            effective_depth > 0 and 0 < index <= effective_depth
            for index in range(4)
        ]
        model.layers[0].reshard()

        if effective_depth == 0:
            # The generic prefetch path fully unshards the immediate next module.
            model.layers[1].reshard()
            return

        # Advancing to layer 1 refills exactly one newly exposed window slot.
        model.layers[1].unshard(async_op=True)
        assert any(
            param_group.state.pending_weights
            for param_group in model.layers[effective_depth + 1]._fsdp_param_groups
        )
        model.layers[1].reshard()

    def test_parameter_group_batches_module_unshard(self, monkeypatch):
        """One module unshard batches all compatible V2 parameter groups."""
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import DataParallelBuffer

        model = MultimodalModel(hidden=16).to(_device())
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
        fully_shard(
            model,
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
        )
        assert len(model._fsdp_param_groups) == 2

        calls = []
        redistribute_buffers = DataParallelBuffer.redistribute_buffers

        def capture_redistribute_buffers(buffers, target_placements, **kwargs):
            calls.append(tuple(buffers))
            return redistribute_buffers(buffers, target_placements, **kwargs)

        monkeypatch.setattr(
            DataParallelBuffer,
            "redistribute_buffers",
            capture_redistribute_buffers,
        )
        try:
            model.unshard()
            assert len(calls) == 1
            assert len(calls[0]) == 2
        finally:
            model.reshard()

    @pytest.mark.parametrize("sharding_strategy", ["no_shard", "optim_grads_params"])
    @pytest.mark.parametrize("use_decoupled_grad", [False, True])
    def test_parameter_group_full_iteration_gradients_are_stable(
        self, sharding_strategy, use_decoupled_grad
    ):
        """Full-iteration mode preserves graph-visible gradient objects and storage."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        fully_shard(
            model,
            sharding_strategy=sharding_strategy,
            mp_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.float32,
                main_grads_dtype=torch.float32,
                use_decoupled_grad=use_decoupled_grad,
            ),
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=True,
            enable_full_iteration_cuda_graph=True,
        )

        def backward():
            model.set_is_last_backward(True)
            _forward_backward(model, torch.randn(2, 16, device=_device()))
            model.finish_grad_sync()

        backward()
        gradient_state = []
        for param_group in model._fsdp_param_groups:
            assert param_group.state.grad_phase is GradientPhase.READY
            assert param_group.grad_buffer.data is not None
            gradient_state.append(
                (
                    param_group,
                    param_group.grad_buffer.data.data_ptr(),
                    tuple(param_group.optimizer_grads),
                )
            )

        model.zero_grad(set_to_none=True)
        for param_group, data_ptr, optimizer_grads in gradient_state:
            assert param_group.state.grad_phase is GradientPhase.EMPTY
            assert param_group.grad_buffer.data.data_ptr() == data_ptr
            assert all(
                current is previous
                for current, previous in zip(param_group.optimizer_grads, optimizer_grads)
            )
            assert torch.count_nonzero(param_group.grad_buffer.data) == 0
            for optimizer_param, optimizer_grad in zip(
                param_group.optimizer_params, optimizer_grads
            ):
                if optimizer_grad is None:
                    continue
                installed_grad = (
                    optimizer_param.decoupled_grad if use_decoupled_grad else optimizer_param.grad
                )
                assert installed_grad is optimizer_grad
                assert optimizer_param._mfsdp_keep_grad_for_cuda_graph

        backward()
        for param_group, data_ptr, optimizer_grads in gradient_state:
            assert param_group.state.grad_phase is GradientPhase.READY
            assert param_group.grad_buffer.data.data_ptr() == data_ptr
            assert all(
                current is previous
                for current, previous in zip(param_group.optimizer_grads, optimizer_grads)
            )

    @pytest.mark.parametrize("outer_dp_sharding_strategy", ["no_shard", "optim"])
    def test_parameter_group_full_iteration_hsdp(self, outer_dp_sharding_strategy):
        """Full-iteration gradient identity is stable for both HSDP final layouts."""
        model = SimpleMLP(16).to(_device())
        fully_shard(
            model,
            mesh=_build_hsdp_mesh(),
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy=outer_dp_sharding_strategy,
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=True,
            enable_full_iteration_cuda_graph=True,
        )

        model.set_is_last_backward(True)
        _forward_backward(model, torch.randn(2, 16, device=_device()))
        model.finish_grad_sync()
        gradient_state = [
            (param_group.grad_buffer.data.data_ptr(), tuple(param_group.optimizer_grads))
            for param_group in model._fsdp_param_groups
        ]

        model.zero_grad(set_to_none=True)
        for param_group, (data_ptr, optimizer_grads) in zip(
            model._fsdp_param_groups, gradient_state
        ):
            assert param_group.grad_buffer.data.data_ptr() == data_ptr
            assert all(
                current is previous
                for current, previous in zip(param_group.optimizer_grads, optimizer_grads)
            )
            assert torch.count_nonzero(param_group.grad_buffer.data) == 0

    def test_parameter_group_full_iteration_capture_and_replay(self):
        """The production full-iteration wrapper captures and replays V2 FSDP."""
        if _world_size() < 2:
            pytest.skip("Full-iteration FSDP capture requires at least two ranks")

        from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader

        FullCudaGraphWrapper.curr_iteration = {"training": 0, "validation": 0}
        FullCudaGraphWrapper.cuda_graph = {"training": None, "validation": None}
        FullCudaGraphWrapper.result = {"training": None, "validation": None}
        StaticBufferLoader.static_buffers = {"training": [], "validation": []}

        torch.manual_seed(42)
        model = SimpleMLP(4, bias=False).to(_device())
        fully_shard(
            model,
            sharding_strategy="optim_grads_params",
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=True,
            enable_full_iteration_cuda_graph=True,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)

        def forward_backward_func(*, model, data_iterator, **_):
            batch = next(data_iterator[0])
            output = model[0](batch["input"])
            loss = output.float().square().mean()
            loss.backward()
            return loss.detach()

        wrapper = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=1)
        losses = []
        try:
            for _ in range(4):
                model.zero_grad(set_to_none=True)
                loss = wrapper(
                    model=[model],
                    data_iterator=[iter([{"input": torch.ones(2, 4, device=_device())}])],
                    num_microbatches=1,
                    seq_length=None,
                    forward_only=False,
                )
                model.finish_grad_sync()
                optimizer.step()
                losses.append(loss.clone())

            assert FullCudaGraphWrapper.cuda_graph["training"] is not None
            loss_values = torch.stack(losses).cpu()
            assert torch.isfinite(loss_values).all()
            assert loss_values[-1] < loss_values[0]
        finally:
            torch.cuda.synchronize()
            wrapper.reset_cuda_graph()
            StaticBufferLoader.static_buffers = {"training": [], "validation": []}

    def test_no_shard_forward_backward_finish_grad_sync(self):
        """no_shard keeps full replicated buffers and all-reduces at grad sync."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model, sharding_strategy="no_shard", enable_async_reduce_grad=False)

        x = torch.randn(2, 64, device=_device())
        model.set_is_last_backward(True)
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss)), "Loss is NaN"
        model.finish_grad_sync()

        for param_group in model._fsdp_param_groups:
            assert param_group.weight_buffer.placements == [Placement.REPLICATE]
            assert param_group.grad_buffer.placements == [Placement.REPLICATE]
            for dist_grad in param_group.optimizer_grads:
                if dist_grad is None:
                    continue
                local_grad = dist_grad._local_tensor
                gathered = [torch.empty_like(local_grad) for _ in range(_world_size())]
                torch.distributed.all_gather(gathered, local_grad)
                for replica in gathered:
                    torch.testing.assert_close(replica, local_grad)

    @pytest.mark.parametrize(
        "enable_unshard_prefetch,enable_async_reduce_grad",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    def test_feature_flags(self, enable_unshard_prefetch, enable_async_reduce_grad):
        """All combinations of overlap flags should work."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(
            model,
            enable_unshard_prefetch=enable_unshard_prefetch,
            enable_async_reduce_grad=enable_async_reduce_grad,
        )
        x = torch.randn(2, 64, device=_device())
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(torch.tensor(loss.item()))

    def test_unshard_coalescing_keeps_mixed_dtypes_separate(self, monkeypatch):
        """Coalesced all-gathers should not group buffers with different dtypes."""
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import dp_buffer as dp_buffer_mod

        torch.manual_seed(42)
        model = MixedDtypeBuffers(16).to(_device())
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        collective_dtypes = []
        coalesced_dtype_runs = []
        active_dtype_runs = []
        original_all_gather = torch.distributed.all_gather_into_tensor
        original_coalescing_manager = dp_buffer_mod._coalescing_manager
        original_redistribute = dp_buffer_mod.DataParallelBuffer.redistribute

        @contextmanager
        def capture_coalescing_manager(group, *args, **kwargs):
            dtype_run = []
            with original_coalescing_manager(group, *args, **kwargs) as event:
                active_dtype_runs.append(dtype_run)
                try:
                    yield event
                finally:
                    active_dtype_runs.pop()
                    coalesced_dtype_runs.append(tuple(dtype_run))

        def capture_redistribute(buffer, *args, **kwargs):
            if active_dtype_runs:
                active_dtype_runs[-1].append(buffer.dtype)
            return original_redistribute(buffer, *args, **kwargs)

        def capture_all_gather(output_tensor, input_tensor, *args, **kwargs):
            collective_dtypes.append(input_tensor.dtype)
            return original_all_gather(output_tensor, input_tensor, *args, **kwargs)

        monkeypatch.setattr(dp_buffer_mod, "_coalescing_manager", capture_coalescing_manager)
        monkeypatch.setattr(dp_buffer_mod.DataParallelBuffer, "redistribute", capture_redistribute)
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", capture_all_gather)

        try:
            model.unshard(async_op=True)
            assert set(collective_dtypes) == {torch.float32, torch.bfloat16}
            assert all(len(set(dtype_run)) == 1 for dtype_run in coalesced_dtype_runs)
        finally:
            model.reshard()

    def test_module_unshard_delegates_weight_ownership_to_param_group(self, monkeypatch):
        """The module should schedule unshard without selecting or binding weight buffers."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        delegated_groups = []
        original_unshard = ParameterGroup.unshard_weights

        def capture_unshard(param_groups, **kwargs):
            delegated_groups.append((tuple(param_groups), kwargs.copy()))
            return original_unshard(param_groups, **kwargs)

        monkeypatch.setattr(ParameterGroup, "unshard_weights", staticmethod(capture_unshard))

        try:
            model.unshard(async_op=False)
            assert delegated_groups == [
                (
                    tuple(model._fsdp_param_groups),
                    {
                        "streams": (torch.cuda.current_stream(),),
                        "async_op": False,
                    },
                )
            ]
        finally:
            model.reshard()

    @pytest.mark.parametrize("outer_strategy", ["no_shard", "optim"])
    def test_weight_unshard_coalesces_outer_before_inner(self, monkeypatch, outer_strategy):
        """Outer runs should finish before inner AGs; no_shard outer is a no-op."""
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2 import dp_buffer as dp_buffer_mod
        from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.dp_buffer import DataParallelBuffer

        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        model.fc.bias.requires_grad_(False)
        fully_shard(
            model,
            mesh=_build_hsdp_mesh(),
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy=outer_strategy,
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=False,
        )

        original_redistribute = DataParallelBuffer.redistribute
        original_coalescing_manager = dp_buffer_mod._coalescing_manager
        original_all_gather = torch.distributed.all_gather_into_tensor
        manager_groups = []
        active_manager_groups = []
        unshard_calls = []
        collective_groups = []
        if outer_strategy == "optim":
            for param_group in model._fsdp_param_groups:
                param_group.refresh_model_weight()

        owned_weight_buffers = [
            param_group.weight_buffer for param_group in model._fsdp_param_groups
        ]
        assert len(owned_weight_buffers) == 2
        first_buffer, second_buffer = owned_weight_buffers
        assert first_buffer.mesh is second_buffer.mesh
        assert first_buffer.dtype == second_buffer.dtype
        assert first_buffer.device == second_buffer.device
        outer_dp_group = first_buffer.mesh.get_group(mesh_dim=0)
        inner_dp_group = first_buffer.mesh.get_group(mesh_dim=1)

        @contextmanager
        def capture_coalescing_manager(group, *args, **kwargs):
            manager_groups.append(group)
            with original_coalescing_manager(group, *args, **kwargs) as event:
                active_manager_groups.append(group)
                try:
                    yield event
                finally:
                    active_manager_groups.pop()

        def capture_redistribute(buffer, *args, **kwargs):
            active_group = active_manager_groups[-1] if active_manager_groups else None
            comm_dim = 0 if active_group is outer_dp_group else 1
            unshard_calls.append((id(buffer.buffer_index), comm_dim, active_group))
            return original_redistribute(buffer, *args, **kwargs)

        def capture_all_gather(*args, **kwargs):
            collective_groups.append(kwargs["group"])
            return original_all_gather(*args, **kwargs)

        monkeypatch.setattr(DataParallelBuffer, "redistribute", capture_redistribute)
        monkeypatch.setattr(dp_buffer_mod, "_coalescing_manager", capture_coalescing_manager)
        monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", capture_all_gather)

        try:
            model.unshard(async_op=True)
            expected_manager_groups = (
                [outer_dp_group, inner_dp_group] if outer_strategy == "optim" else [inner_dp_group]
            )
            assert len(manager_groups) == len(expected_manager_groups)
            assert all(
                actual is expected
                for actual, expected in zip(manager_groups, expected_manager_groups)
            )
            expected_unshard_calls = []
            if outer_strategy == "optim":
                expected_unshard_calls.extend(
                    [
                        (id(first_buffer.buffer_index), 0, outer_dp_group),
                        (id(second_buffer.buffer_index), 0, outer_dp_group),
                    ]
                )
            expected_unshard_calls.extend(
                [
                    (id(first_buffer.buffer_index), 1, inner_dp_group),
                    (id(second_buffer.buffer_index), 1, inner_dp_group),
                ]
            )
            assert len(unshard_calls) == len(expected_unshard_calls)
            assert all(
                actual_buffer == expected_buffer
                and actual_dim == expected_dim
                and actual_group is expected_group
                for (actual_buffer, actual_dim, actual_group), (
                    expected_buffer,
                    expected_dim,
                    expected_group,
                ) in zip(unshard_calls, expected_unshard_calls)
            )
            expected_collective_groups = (
                [outer_dp_group, outer_dp_group] if outer_strategy == "optim" else []
            )
            expected_collective_groups.extend([inner_dp_group, inner_dp_group])
            assert len(collective_groups) == len(expected_collective_groups)
            assert all(
                actual is expected
                for actual, expected in zip(collective_groups, expected_collective_groups)
            )
            assert all(
                param_group.weights_are_unsharded()
                for param_group in model._fsdp_param_groups
            )
        finally:
            model.reshard()

    def test_outer_optim_refreshes_replica_in_next_forward(self):
        """The last backward lazily refreshes BF16 HSDP replicas before forward."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(device=_device(), dtype=torch.bfloat16)
        fully_shard(
            model,
            mesh=_build_hsdp_mesh(),
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy="optim",
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
        )

        param_group = model._fsdp_param_groups[0]
        model_buffer = param_group.weight_buffer
        main_buffer = param_group.main_weight_buffer
        assert model_buffer.placements == [Placement.REPLICATE, Placement.SHARD]
        assert main_buffer.placements == [Placement.SHARD, Placement.SHARD]
        assert (
            main_buffer.data.untyped_storage().data_ptr()
            == model_buffer.data.untyped_storage().data_ptr()
        )
        from torch.distributed.tensor.placement_types import Shard

        assert all(
            isinstance(placement, Shard) for placement in param_group.optimizer_params[0].placements
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
        x = torch.full((2, 16), _rank() + 1, device=_device(), dtype=torch.bfloat16)

        model.set_is_last_backward(False)
        model(x).float().sum().backward()
        assert not model._fsdp_root_context.model_weight_refresh_pending

        model.set_is_last_backward(True)
        model(x).float().sum().backward()
        model.finish_grad_sync()
        assert param_group.grad_buffer.placements == [Placement.REPLICATE, Placement.SHARD]
        optimizer_grad_view = param_group.grad_buffer.view([Placement.SHARD, Placement.SHARD])
        assert optimizer_grad_view.data.numel() * _world_size() == (
            param_group.grad_buffer.buffer_index.bucket_meta.size
        )

        ctx = model._fsdp_root_context
        assert ctx.model_weight_refresh_pending

        optimizer.step()
        # No optimizer integration performed an explicit model-weight copy.
        assert ctx.model_weight_refresh_pending
        assert model_buffer.placements == [Placement.REPLICATE, Placement.SHARD]

        # The normal pre-forward hook selects the direct model-weight SHARD view
        # before unshard, which refreshes the outer replicas exactly once.
        model(torch.zeros_like(x))
        assert not ctx.model_weight_refresh_pending
        assert param_group.state.weight_valid == tuple(model_buffer.placements)
        assert param_group.compute_weight() is None

        outer_replicas = [
            torch.empty_like(model_buffer.data)
            for _ in range(
                torch.distributed.get_world_size(model_buffer.mesh.get_group(mesh_dim=0))
            )
        ]
        torch.distributed.all_gather(
            outer_replicas, model_buffer.data, group=model_buffer.mesh.get_group(mesh_dim=0)
        )
        assert all(torch.equal(replica, outer_replicas[0]) for replica in outer_replicas[1:])

    def test_skipped_prefetch_waits_before_reshard(self, monkeypatch):
        """A skipped prefetched module must join its AG before freeing buffers."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=1).to(_device())
        layer = model.layers[0]
        fully_shard(layer, enable_unshard_prefetch=True, enable_async_reduce_grad=False)
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        ctx = model._fsdp_root_context
        model.unshard(async_op=True)
        model.reshard()

        real_event = ctx.unshard_done_events[id(layer)]
        assert real_event is not None
        real_event.synchronize()

        order = []

        class CompletedEvent:
            def wait(self):
                order.append("wait")

        ctx.unshard_done_events[id(layer)] = CompletedEvent()
        for param_group in layer._fsdp_param_groups:
            original_reshard = param_group.reshard_weight

            def capture_reshard(*, _original=original_reshard):
                order.append("reshard")
                return _original()

            monkeypatch.setattr(param_group, "reshard_weight", capture_reshard)

        try:
            layer.reshard()
            assert order
            assert order[0] == "wait"
            assert ctx.unshard_done_events[id(layer)] is None
        finally:
            model.reshard()
            layer.reshard()


# ------------------------------------------------------------------ #
#  2. Scenarios — LLM-style nesting (embedding + layers + lm_head)
# ------------------------------------------------------------------ #


class TestLLMScenario:
    def test_llm_full_shard_root(self):
        """Shard the root TinyLLM — all params go into one FSDP module."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=128, hidden=64, num_layers=2).to(_device())
        fully_shard(model)
        assert _count_fsdp_modules(model) == 1
        _assert_dtensor_params(model)

        x = torch.randint(0, 128, (4, 8), device=_device())
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss))

    def test_llm_per_layer_shard(self):
        """Shard each transformer layer individually (typical FSDP setup)."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=128, hidden=64, num_layers=2).to(_device())
        # Shard each child layer separately
        for layer in model.layers:
            fully_shard(layer)
        # Shard the root (embedding + lm_head + norm covered)
        fully_shard(model)

        assert _count_fsdp_modules(model) == 3  # 2 layers + root
        _assert_dtensor_params(model)

        x = torch.randint(0, 128, (4, 8), device=_device())
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss))


# ------------------------------------------------------------------ #
#  3. Multimodal scenario — separate encoders + partial freezing
# ------------------------------------------------------------------ #


class TestMultimodalScenario:
    def test_frozen_vision_encoder(self):
        """Freeze vision encoder params; they should NOT be sharded (no grad)."""
        torch.manual_seed(42)
        model = MultimodalModel(hidden=64).to(_device())
        # Freeze vision encoder
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)

        x_img = torch.randn(2, 64, device=_device(), requires_grad=True)
        x_txt = torch.randn(2, 64, device=_device(), requires_grad=True)
        out = model(x_img, x_txt)
        loss = out.sum()
        loss.backward()

        # Frozen params should have no grad after reduce_grad
        for name, p in model.named_parameters():
            if "vision_encoder" in name:
                assert not p.requires_grad, f"Frozen param {name} should have requires_grad=False"

        assert not torch.isnan(torch.tensor(loss.item()))

    def test_frozen_params_in_own_group(self):
        """Frozen params are included but grouped separately (different requires_grad)."""
        torch.manual_seed(42)
        model = MultimodalModel(hidden=64).to(_device())
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)

        # Frozen params should be in a group with requires_grad=False
        has_frozen_group = False
        for param_group in model._fsdp_param_groups:
            if not param_group.requires_grad:
                has_frozen_group = True
                for p in param_group.params:
                    assert (
                        not p.requires_grad
                    ), "Param in frozen group should have requires_grad=False"
        assert has_frozen_group, "Frozen params should be in their own param group"

    def test_mixed_frozen_and_trainable(self):
        """Some parts frozen, some trainable — all sharded together."""
        torch.manual_seed(42)
        model = MultimodalModel(hidden=64).to(_device())
        # Freeze only vision, text and fusion stay trainable
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)

        x_img = torch.randn(2, 64, device=_device(), requires_grad=True)
        x_txt = torch.randn(2, 64, device=_device(), requires_grad=True)

        # Should run without error
        out = model(x_img, x_txt)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  4. Nested FSDP — expert-in-layer (EDP pattern)
# ------------------------------------------------------------------ #


class TestNestedFSDP:
    def test_nested_expert_in_layer(self):
        """Shard experts inside layer, then shard layer, then root — EDP pattern."""
        torch.manual_seed(42)
        device = _device()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = MOETransformerLayer(64, 128)
                self.head = nn.Linear(64, 10)

            def forward(self, x):
                return self.head(self.layer(x))

        model = Model().to(device)

        # Step 1: shard the expert (nested FSDP)
        model.layer.experts = fully_shard(model.layer.experts)
        # Step 2: shard the layer (will detect nested FSDP)
        model.layer = fully_shard(model.layer)
        # Step 3: shard the root
        model = fully_shard(model)

        assert _count_fsdp_modules(model) == 3  # experts, layer, root

        x = torch.randn(2, 64, device=device, requires_grad=True)
        loss = _forward_backward(model, x)
        assert not torch.isnan(torch.tensor(loss))

    def test_nested_ignored_params_are_skipped(self):
        """Nested FSDP module params must be in the parent's ignored_params."""
        torch.manual_seed(42)
        device = _device()
        model = MOETransformerLayer(64, 128).to(device)

        expert = fully_shard(model.experts)
        model.experts = expert  # rebind (fully_shard returns same object)
        model = fully_shard(model)

        # The outer FSDP (model) should have a parameter group that does NOT
        # include the inner FSDP's (expert) params
        expert_param_ids = set(id(p) for p in expert.parameters())
        for _, param_group in model._named_param_groups:
            for p in param_group.params:
                assert (
                    id(p) not in expert_param_ids
                ), "Nested FSDP param leaked into parent param group"

    def test_nested_forward_backward(self):
        """Nested FSDP forward+backward produces correct loss pattern."""
        torch.manual_seed(42)
        device = _device()
        model = MOETransformerLayer(64, 128).to(device)

        model.experts = fully_shard(model.experts)
        model = fully_shard(model)

        # Run twice — second pass should work correctly (buffer reuse)
        for _ in range(2):
            x = torch.randn(2, 64, device=device, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  5. Mixed precision policies
# ------------------------------------------------------------------ #


class TestMixedPrecision:
    def test_main_params_fp32(self):
        """With fp32 main params, main_weight_buffer should be created."""
        torch.manual_seed(42)
        mp_policy = MixedPrecisionPolicy(main_params_dtype=torch.float32)
        model = SimpleMLP(64).to(_device()).bfloat16()
        fully_shard(model, mp_policy=mp_policy)

        # Verify the separate main-weight storage is fp32.
        for param_group in model._fsdp_param_groups:
            assert (
                param_group.main_weight_buffer.dtype == torch.float32
            ), f"Expected fp32 main weight buffer, got {param_group.main_weight_buffer.dtype}"

    def test_main_params_none(self):
        """Without a main dtype, optimizer weights alias model-weight storage."""
        torch.manual_seed(42)
        mp_policy = MixedPrecisionPolicy(main_params_dtype=None)
        model = SimpleMLP(64).to(_device())
        fully_shard(model, mp_policy=mp_policy)

        for param_group in model._fsdp_param_groups:
            assert (
                param_group.main_weight_buffer.data.untyped_storage().data_ptr()
                == param_group.weight_buffer.data.untyped_storage().data_ptr()
            )

    def test_fp32_grad_reduce(self):
        """grad_reduce_in_fp32=True should use fp32 gradient communication."""
        torch.manual_seed(42)
        mp_policy = MixedPrecisionPolicy(grad_comm_dtype=torch.float32)
        model = SimpleMLP(64).to(_device()).bfloat16()
        fully_shard(model, mp_policy=mp_policy)

        x = torch.randn(2, 64, device=_device(), dtype=torch.bfloat16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  6. ignored_params
# ------------------------------------------------------------------ #


class TestIgnoredParams:
    def test_ignored_params_excluded_from_groups(self):
        """Params passed as ignored_params should not appear in FSDP groups."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        # Pre-identify the param to ignore before wrapping
        ignored = {model.fc.weight}
        fully_shard(model, ignored_params=ignored)

        for param_group in model._fsdp_param_groups:
            for p in param_group.params:
                assert p is not model.fc.weight, "Ignored param leaked into group"

    def test_ignored_param_stays_on_module(self):
        """Ignored param should remain as a regular nn.Parameter on the module."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        original_weight = model.fc.weight
        ignored = {model.fc.weight}
        fully_shard(model, ignored_params=ignored)

        # After fully_shard and reshard, the ignored weight should still be
        # the original nn.Parameter (not a DTensor) on the module
        from torch.distributed.tensor import DTensor

        assert not isinstance(
            model.fc.weight, DTensor
        ), "Ignored param should not be converted to DTensor"
        assert model.fc.weight is original_weight, "Ignored param identity changed"


# ------------------------------------------------------------------ #
#  7. Forward/backward lifecycle correctness
# ------------------------------------------------------------------ #


class TestLifecycle:
    @pytest.mark.parametrize("sharding_strategy", ["no_shard", "optim_grads_params"])
    def test_cpu_offload_rebinds_persistent_parameter_group_views(self, sharding_strategy):
        """CPU offload and automatic reload preserve optimizer DTensor identities."""
        model = fully_shard(
            SimpleMLP(16).to(_device(), dtype=torch.bfloat16),
            mp_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.float32,
                main_grads_dtype=torch.float32,
            ),
            sharding_strategy=sharding_strategy,
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
        )
        model.set_is_last_backward(True)
        model(
            torch.randn(2, 16, device=_device(), dtype=torch.bfloat16)
        ).float().square().mean().backward()
        model.finish_grad_sync()

        optimizer_ids = [
            id(optimizer_param)
            for param_group in model._fsdp_param_groups
            for optimizer_param in param_group.optimizer_params
        ]
        result = model.offload_to_cpu(pin_memory=True)
        assert result["offloaded_bytes"] > 0
        for param_group in model._fsdp_param_groups:
            assert all(
                buffer.data is None or buffer.data.device.type == "cpu"
                for buffer in (
                    param_group.weight_buffer,
                    param_group.main_weight_buffer,
                    param_group.grad_buffer,
                )
            )
            assert all(
                optimizer_param._local_tensor.device.type == "cpu"
                for optimizer_param in param_group.optimizer_params
            )
            if param_group.accumulates_full_grad:
                assert param_group.state.full_grad is not None
                assert param_group.state.full_grad.data.device.type == "cpu"
                assert (
                    param_group.state.full_grad.data.data_ptr()
                    == param_group.grad_buffer.data.data_ptr()
                )

        model.unshard()
        assert optimizer_ids == [
            id(optimizer_param)
            for param_group in model._fsdp_param_groups
            for optimizer_param in param_group.optimizer_params
        ]
        for param_group in model._fsdp_param_groups:
            assert all(
                buffer.data is None or buffer.data.device.type == "cuda"
                for buffer in (
                    param_group.weight_buffer,
                    param_group.main_weight_buffer,
                    param_group.grad_buffer,
                )
            )
            if param_group.accumulates_full_grad:
                assert param_group.state.full_grad is not None
                assert param_group.state.full_grad.data.device.type == "cuda"
                assert (
                    param_group.state.full_grad.data.data_ptr()
                    == param_group.grad_buffer.data.data_ptr()
                )
            assert all(
                optimizer_param._local_tensor.device.type == "cuda"
                for optimizer_param in param_group.optimizer_params
            )
        model.reshard()

    def test_external_backward_callbacks_finalize_parameter_group_gradients(self):
        """Delayed-wgrad callers may explicitly finalize gradients after backward."""
        model = fully_shard(
            SimpleMLP(16).to(_device()),
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
            skip_backward_callback=True,
            skip_final_backward_callback=True,
        )
        model.set_is_last_backward(True)

        model(torch.randn(2, 16, device=_device())).float().square().mean().backward()
        for param_group in model._fsdp_param_groups:
            assert param_group.state.grad_phase is GradientPhase.EMPTY

        mfsdp_post_backward_hook(model)
        mfsdp_post_backward_final_callback(model)

        for param_group in model._fsdp_param_groups:
            assert param_group.state.grad_phase is GradientPhase.READY
            assert all(
                optimizer_grad is not None
                for optimizer_param, optimizer_grad in zip(
                    param_group.optimizer_params, param_group.optimizer_grads
                )
                if optimizer_param.requires_grad
            )

    def test_trace_pool_plans_and_reuses_parameter_group_scratch(self):
        """TracePoolAllocator plans the new parameter-group scratch lifecycle."""
        torch.manual_seed(42)
        model = fully_shard(
            SimpleMLP(16).to(_device()),
            enable_trace_pool=True,
            enable_unshard_prefetch=False,
            enable_async_reduce_grad=False,
        )
        allocator = model._fsdp_root_context.bucket_allocator
        assert isinstance(allocator, TracePoolAllocator)
        assert allocator.phase == "trace"

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scratch_addresses = None
        for _ in range(2):
            model(torch.randn(2, 16, device=_device())).float().square().mean().backward()
            model.finish_grad_sync()
            optimizer.step()
            model.zero_grad(set_to_none=True)

            assert allocator.phase == "optimized"
            current_addresses = {
                key: view.data_ptr() for key, view in allocator._key_to_view.items()
            }
            if scratch_addresses is None:
                scratch_addresses = current_addresses
            else:
                assert current_addresses == scratch_addresses

    def test_root_grad_release_skips_full_iteration_cuda_graph(self, monkeypatch):
        """Full-iteration graphs own stable grad storage and zeroing."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        model = fully_shard(model, enable_full_iteration_cuda_graph=True)
        param_group = model._fsdp_param_groups[0]
        zero_grad_calls = []

        def capture_zero_grad(*args, **kwargs):
            zero_grad_calls.append((args, kwargs))

        monkeypatch.setattr(param_group, "zero_grad", capture_zero_grad)

        model._release_grad_storage_if_unused()

        assert zero_grad_calls == []

    def test_root_grad_release_keeps_live_accumulation(self):
        """A live optimizer-facing grad prevents its storage release."""
        torch.manual_seed(42)
        model = SimpleMLP(16).to(_device())
        # Keep an optimizer-facing gradient on every rank so the test's
        # liveness precondition does not depend on the local shard being empty.
        model = fully_shard(model, sharding_strategy="no_shard")

        model(torch.randn(2, 16, device=_device())).float().square().mean().backward()
        model.finish_grad_sync()
        param_group = model._fsdp_param_groups[0]
        assert param_group.grad_buffer.data is not None
        assert any(
            getattr(dist_param, "grad", None) is not None
            or getattr(dist_param, "decoupled_grad", None) is not None
            for dist_param in param_group.optimizer_params
        )

        model._release_grad_storage_if_unused()

        assert param_group.grad_buffer.data is not None

    def test_root_forward_releases_unused_grad_storage_per_group(self):
        """The root sweep releases eligible groups without a global liveness check."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=2).to(_device())
        for index, layer in enumerate(model.layers):
            model.layers[index] = fully_shard(layer, sharding_strategy="no_shard")
        model = fully_shard(model, sharding_strategy="no_shard")
        param_groups = [
            param_group
            for module in model._get_fsdp_modules(recursive=True)
            for param_group in module._fsdp_param_groups
        ]
        assert len(param_groups) > 1
        for param_group in param_groups:
            param_group.prepare_gradient_storage()

        live_group = param_groups[0]
        live_group._install_optimizer_grads()
        released_groups = param_groups[1:]
        assert any(
            getattr(dist_param, "grad", None) is not None
            for dist_param in live_group.optimizer_params
        )

        model(torch.randint(0, 32, (2, 4), device=_device()))

        assert live_group.grad_buffer.data is not None
        assert all(param_group.grad_buffer.data is None for param_group in released_groups)

    def test_root_forward_releases_optimizer_cleared_grad_storage_before_unshard(self, monkeypatch):
        """Plain optimizer zero-grad must not overlap stale grads with next unshard."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=2).to(_device())
        # Replicated gradients make the storage assertions valid on every
        # world size used by CI, including ranks that would own empty shards.
        for index, layer in enumerate(model.layers):
            model.layers[index] = fully_shard(layer, sharding_strategy="no_shard")
        model = fully_shard(model, sharding_strategy="no_shard")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randint(0, 32, (2, 4), device=_device())
        model(x).float().square().mean().backward()
        model.finish_grad_sync()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        param_groups = [
            param_group
            for module in model._get_fsdp_modules(recursive=True)
            for param_group in module._fsdp_param_groups
        ]
        assert any(
            param_group.grad_buffer.data is not None for param_group in param_groups
        )
        assert all(
            getattr(dist_param, "grad", None) is None
            and getattr(dist_param, "decoupled_grad", None) is None
            for param_group in param_groups
            for dist_param in param_group.optimizer_params
        )

        observed_at_root_unshard = []
        original_unshard = model.unshard

        def capture_root_unshard(*args, **kwargs):
            observed_at_root_unshard.append(
                [
                    param_group.grad_buffer.data is None
                    for param_group in param_groups
                ]
            )
            return original_unshard(*args, **kwargs)

        monkeypatch.setattr(model, "unshard", capture_root_unshard)
        model(torch.randint(0, 32, (2, 4), device=_device()))

        assert observed_at_root_unshard
        assert all(observed_at_root_unshard[0])

    def test_fused_adam_reuses_dist_grad_wrappers_across_steps(self):
        """Rebound DTensor grads retain optimizer-compatible shape and identity."""
        torch.manual_seed(42)
        # Wrapper identity is the behavior under test. Use replicated grads so
        # every CI rank owns wrappers and follows the same iteration path.
        model = fully_shard(SimpleMLP(16).to(_device()), sharding_strategy="no_shard")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
        x = torch.randn(2, 16, device=_device())
        wrapper_ids = None

        for iteration in range(3):
            loss = model(x).float().square().mean()
            loss.backward()
            model.finish_grad_sync()

            param_group = model._fsdp_param_groups[0]
            live_grads = [grad for grad in param_group.optimizer_grads if grad is not None]
            assert live_grads
            if wrapper_ids is None:
                wrapper_ids = [id(grad) for grad in live_grads]
            else:
                assert [id(grad) for grad in live_grads] == wrapper_ids
            for dist_param, dist_grad in zip(param_group.optimizer_params, param_group.optimizer_grads):
                if dist_grad is not None:
                    assert dist_grad._local_tensor.shape == dist_param._local_tensor.shape

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            assert torch.isfinite(loss)

            if iteration < 2:
                # Plain optimizer zero-grad leaves release to the next root
                # forward. The following iteration exercises detach + rebind.
                assert param_group.grad_buffer.data is not None

        assert all(torch.isfinite(param._local_tensor).all() for param in model.parameters())

    def test_params_unsharded_during_forward(self):
        """During forward, model parameters should be in unsharded state (full tensors)."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())

        captured_shapes = []

        def hook(module, inp):
            # At this point, the pre-forward hook has already unsharded
            for name, p in module.named_parameters():
                captured_shapes.append((name, p.data.shape))

        model.register_forward_pre_hook(hook)
        fully_shard(model)

        # Before forward, params should be DTensors (reshard called at init)
        from torch.distributed.tensor import DTensor

        for _, p in model.named_parameters():
            assert isinstance(p, DTensor), "Params should be DTensors after init reshard"

        x = torch.randn(2, 64, device=_device())
        model(x)

        # Inside the forward pre-hook, params should be full tensors
        for name, shape in captured_shapes:
            assert shape == torch.Size([64, 64]) or shape == torch.Size(
                [64]
            ), f"Param {name} has wrong shape during forward: {shape}"

    def test_params_resharded_after_forward(self):
        """After forward, model parameters should be resharded back to DTensors."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)

        from torch.distributed.tensor import DTensor

        x = torch.randn(2, 64, device=_device())
        model(x)

        for _, p in model.named_parameters():
            assert isinstance(p, DTensor), "Params should be DTensors after forward reshard"

    def test_params_unsharded_during_backward(self):
        """During backward, model parameters should be unsharded."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)

        from torch.distributed.tensor import DTensor

        captured_dtensor = []

        def grad_hook(grad):
            for _, p in model.named_parameters():
                captured_dtensor.append(isinstance(p, DTensor))

        x = torch.randn(2, 64, device=_device(), requires_grad=True)
        out = model(x)
        out.register_hook(grad_hook)
        loss = out.sum()
        loss.backward()

        # During backward (grad_hook fires during backward pass), params should NOT be DTensor
        for is_dt in captured_dtensor:
            assert not is_dt, "Params should be unsharded during backward"

    def test_params_resharded_after_backward(self):
        """After full backward pass, params should be DTensors again."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)

        from torch.distributed.tensor import DTensor

        x = torch.randn(2, 64, device=_device(), requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for _, p in model.named_parameters():
            assert isinstance(p, DTensor), "Params should be DTensors after backward reshard"


# ------------------------------------------------------------------ #
#  8. Activation checkpointing
# ------------------------------------------------------------------ #


class MLPWithCheckpointing(nn.Module):
    """A multi-layer MLP that supports activation checkpointing on its blocks."""

    def __init__(self, hidden=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                for _ in range(num_layers)
            ]
        )
        self._use_activation_checkpointing = False

    def forward(self, x):
        for layer in self.layers:
            if self._use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

    def enable_activation_checkpointing(self):
        self._use_activation_checkpointing = True


class LargePerLayerModel(nn.Module):
    """Multi-layer model with individually wrapped FSDP layers and optional
    activation checkpointing support."""

    def __init__(self, hidden=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestActivationCheckpointing:
    def test_recompute_successor_uses_updated_weight_after_optimizer_step(self):
        """A recompute-prefetched successor must not reuse pre-step weights.

        Layer 1 finishes backward before layer 0 is recomputed. A normal
        forward-prefetch from that recompute can incorrectly resurrect layer
        1's full model-weight buffer after its post-backward reshard. Under
        outer ``no_shard``, copying the optimizer's FP32 main shard updates
        only persistent BF16 storage, so the resurrected full buffer would be
        stale on the next forward.

        Use per-layer FSDP units (and an FSDP root), but no nested expert unit:
        this isolates successor prefetch from the separate nested post-forward
        lifecycle path.
        """
        torch.manual_seed(42)
        device = _device()
        mesh = _build_hsdp_mesh()

        class TwoLayerCheckpointModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [
                        nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 32))
                        for _ in range(2)
                    ]
                )

            def forward(self, x):
                for layer in self.layers:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                return x

        model = TwoLayerCheckpointModel().to(device=device, dtype=torch.bfloat16)
        shard_kwargs = dict(
            mesh=mesh,
            sharding_strategy="optim_grads_params",
            outer_dp_sharding_strategy="no_shard",
            mp_policy=MixedPrecisionPolicy(
                main_params_dtype=torch.float32,
                main_grads_dtype=torch.float32,
                grad_comm_dtype=torch.float32,
            ),
            enable_unshard_prefetch=True,
            enable_async_reduce_grad=True,
        )
        for index, layer in enumerate(model.layers):
            model.layers[index] = fully_shard(layer, **shard_kwargs)
        model = fully_shard(model, **shard_kwargs)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

        successor = model.layers[1]
        param_group = successor._fsdp_param_groups[0]
        model_buffer = param_group.weight_buffer
        main_buffer = param_group.main_weight_buffer
        assert main_buffer is not None
        expected_placements = [Placement.REPLICATE, Placement.SHARD]
        assert model_buffer.placements == expected_placements
        assert main_buffer.placements == expected_placements

        model.set_is_last_backward(True)
        x = torch.randn(4, 32, device=device, dtype=torch.bfloat16, requires_grad=True)
        with torch.utils.checkpoint.set_checkpoint_early_stop(False):
            loss = model(x).float().square().mean()
            loss.backward()
        model.finish_grad_sync()

        main_before = main_buffer.data.detach().clone()
        optimizer.step()
        assert not torch.equal(main_buffer.data, main_before), "SGD did not update successor"
        model._copy_main_weights_to_model_weights()
        assert torch.equal(model_buffer.data, main_buffer.data.to(model_buffer.dtype))

        expected_full = torch.empty(
            model_buffer.buffer_index.bucket_meta.size, dtype=model_buffer.dtype, device=device
        )
        torch.distributed.all_gather_into_tensor(
            expected_full, model_buffer.data, group=model_buffer.mesh.get_group(mesh_dim=1)
        )

        observed_full = []

        def capture_successor_full_buffer(_module, _args):
            compute_buffer = param_group.compute_weight()
            assert compute_buffer is not None
            observed_full.append(compute_buffer.data.detach().clone())

        handle = successor.register_forward_pre_hook(capture_successor_full_buffer)
        try:
            x_next = torch.randn(4, 32, device=device, dtype=torch.bfloat16)
            model(x_next)
        finally:
            handle.remove()

        assert len(observed_full) == 1
        for item_id in range(len(param_group.params)):
            start, end = model_buffer.buffer_index._get_item_global_range(item_id)
            torch.testing.assert_close(
                observed_full[0][start:end], expected_full[start:end], rtol=0, atol=0
            )

    def test_recompute_forward_self_unshard_disables_prefetch(self, monkeypatch):
        """Recompute may unshard itself but must not advance forward prefetch."""
        torch.manual_seed(42)
        model = TinyLLM(vocab=32, hidden=16, num_layers=1).to(_device())
        target = model.layers[0]
        fully_shard(target, enable_unshard_prefetch=True, enable_async_reduce_grad=False)
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=False)

        assert not target._fsdp_state._is_root
        ctx = model._fsdp_root_context
        ctx.backward_phase = True
        ctx.backward_module = id(target)

        calls = []

        def capture_unshard(async_op=False, bwd_pass=False, prefetch=True):
            calls.append((async_op, bwd_pass, prefetch))

        monkeypatch.setattr(target, "unshard", capture_unshard)
        mfsdp_forward_pre_hook(target, (), {})

        assert calls == [(True, True, True), (True, False, False)]

    def test_activation_checkpointing_forward_backward(self):
        """Forward + backward with activation checkpointing should produce finite loss."""
        torch.manual_seed(42)
        device = _device()
        model = MLPWithCheckpointing(hidden=64, num_layers=4).to(device)
        model.enable_activation_checkpointing()

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        _assert_dtensor_params(model)

        x = torch.randn(2, 64, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(torch.tensor(loss.item())), "Loss is NaN"
        assert not torch.isinf(torch.tensor(loss.item())), "Loss is Inf"

    def test_activation_checkpointing_multi_step(self):
        """Multiple forward+backward steps with activation checkpointing should be stable."""
        torch.manual_seed(42)
        device = _device()
        model = MLPWithCheckpointing(hidden=64, num_layers=4).to(device)
        model.enable_activation_checkpointing()

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        losses = []
        for step in range(4):
            torch.manual_seed(step)
            x = torch.randn(2, 64, device=device, requires_grad=True)
            out = model(x)
            loss = out.sum()
            loss.backward()
            losses.append(loss.item())

        for i, loss_val in enumerate(losses):
            assert not torch.isnan(torch.tensor(loss_val)), f"Loss at step {i} is NaN"
            assert not torch.isinf(torch.tensor(loss_val)), f"Loss at step {i} is Inf"

    def test_activation_checkpointing_with_overlap(self):
        """Activation checkpointing should work with unshard_prefetch and async_reduce_grad."""
        torch.manual_seed(42)
        device = _device()
        model = MLPWithCheckpointing(hidden=128, num_layers=4).to(device)
        model.enable_activation_checkpointing()

        for layer in model.layers:
            fully_shard(layer, enable_unshard_prefetch=True, enable_async_reduce_grad=True)
        fully_shard(model, enable_unshard_prefetch=True, enable_async_reduce_grad=True)

        x = torch.randn(2, 128, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(torch.tensor(loss.item()))

    def test_activation_checkpointing_nested_fsdp(self):
        """Activation checkpointing with nested FSDP (expert-in-layer) should work."""
        torch.manual_seed(42)
        device = _device()

        class NestedCheckpointModel(nn.Module):
            def __init__(self, hidden=64):
                super().__init__()
                self.attn = nn.Linear(hidden, hidden)
                self.experts = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                self.norm = nn.LayerNorm(hidden)

            def forward(self, x):
                h = self.attn(x)
                if self._use_activation_checkpointing:
                    h = torch.utils.checkpoint.checkpoint(self.experts, h, use_reentrant=False)
                else:
                    h = self.experts(h)
                return self.norm(h + x)

        model = NestedCheckpointModel(hidden=64).to(device)
        model._use_activation_checkpointing = True
        model.experts = fully_shard(model.experts)
        model = fully_shard(model)

        x = torch.randn(2, 64, device=device, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert not torch.isnan(torch.tensor(loss.item()))

    def test_activation_checkpointing_disabled_vs_enabled_same_loss(self):
        """With same inputs and no parameter updates, checkpointed and non-checkpointed
        forward should produce the same output (checkpoint recompute is numerically transparent)."""
        torch.manual_seed(42)
        device = _device()

        model = MLPWithCheckpointing(hidden=64, num_layers=3).to(device)

        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

        x = torch.randn(2, 64, device=device, requires_grad=True)

        # Forward without activation checkpointing
        model._use_activation_checkpointing = False
        out_no_ckpt = model(x)

        # Forward with activation checkpointing
        torch.manual_seed(42)
        x2 = torch.randn(2, 64, device=device, requires_grad=True)
        model._use_activation_checkpointing = True
        out_ckpt = model(x2)

        assert torch.allclose(
            out_no_ckpt, out_ckpt, atol=1e-5
        ), "Checkpointing changed forward output (same inputs)"

    def test_activation_checkpointing_per_layer_shard_with_ckpt(self):
        """Per-layer FSDP with activation checkpointing on each layer — full training step."""
        torch.manual_seed(42)
        device = _device()
        model = LargePerLayerModel(hidden=256, num_layers=6).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for layer in model.layers:
            fully_shard(layer)

        fully_shard(model)

        # Use checkpoint on every other layer to test mixed use
        def ckpt_forward(x):
            for i, layer in enumerate(model.layers):
                if i % 2 == 0:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
                else:
                    x = layer(x)
            return x

        x = torch.randn(4, 256, device=device, requires_grad=True)
        out = ckpt_forward(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        assert not torch.isnan(torch.tensor(loss.item()))


# ------------------------------------------------------------------ #
#  9. Safety — double-shard rejection
# ------------------------------------------------------------------ #


class TestSafety:
    def test_double_shard_rejected(self):
        """Calling fully_shard on an already-wrapped module should raise ValueError."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        with pytest.raises(ValueError, match="already been fully sharded"):
            fully_shard(model)

    def test_no_params_module_ok(self):
        """fully_shard on a module with no parameters should succeed (no-op)."""
        model = nn.Sequential().to(_device())
        wrapped = fully_shard(model)
        assert isinstance(wrapped, FSDPModule)
        assert _count_fsdp_modules(wrapped) == 1


# ------------------------------------------------------------------ #
# 10. Checkpoint — get_state_dict and preprocess_state_dict_for_uneven_dtensor
# ------------------------------------------------------------------ #


class TestCheckpoint:
    def test_get_state_dict_returns_dicts(self):
        """get_state_dict should return model and optimizer state dicts."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run one step to populate optimizer state
        x = torch.randn(2, 64, device=_device())
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert isinstance(model_sd, dict)
        assert isinstance(opt_sd, dict)
        assert len(model_sd) > 0, "Model state dict should not be empty"

    def test_get_state_dict_nested_fsdp(self):
        """get_state_dict should work with nested FSDP modules."""
        torch.manual_seed(42)
        device = _device()
        model = MOETransformerLayer(64, 128).to(device)
        model.experts = fully_shard(model.experts)
        model = fully_shard(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        x = torch.randn(2, 64, device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert len(model_sd) > 0

    @pytest.mark.skip(reason="Hangs. Debug in progress.")
    def test_preprocess_state_dict_adds_metadata(self):
        """preprocess_state_dict_for_uneven_dtensor should add chunk metadata."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        fully_shard(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.0)

        # Build a raw state dict via torch's state_dict
        sd = torch_get_state_dict(
            model, opt, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        preprocess_state_dict_for_uneven_dtensor(sd)

        # Check that the state dict still contains parameter data
        assert len(sd) > 0

    def test_get_state_dict_strict_all_dtensor(self):
        """get_state_dict should assert all params are DTensors."""
        torch.manual_seed(42)
        model = SimpleMLP(64).to(_device())
        # DON'T call fully_shard — params are NOT DTensors
        optimizer = torch.optim.AdamW(model.parameters())

        with pytest.raises(AssertionError, match="Expected all parameters to be DTensors"):
            get_state_dict(model, optimizer)

    def test_get_state_dict_llm_scenario(self):
        """Full LLM forward-backward-checkpoint cycle should work."""
        torch.manual_seed(42)
        device = _device()
        model = TinyLLM(vocab=128, hidden=64, num_layers=2).to(device)
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, 128, (4, 8), device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert len(model_sd) > 0
        assert len(opt_sd) > 0

    def test_get_state_dict_with_frozen_params(self):
        """get_state_dict should work with mixed frozen/trainable params."""
        torch.manual_seed(42)
        device = _device()
        model = MultimodalModel(hidden=64).to(device)
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
        fully_shard(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        x_img = torch.randn(2, 64, device=device, requires_grad=True)
        x_txt = torch.randn(2, 64, device=device, requires_grad=True)
        out = model(x_img, x_txt)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        model_sd, opt_sd = get_state_dict(model, optimizer)
        assert len(model_sd) > 0

    def test_get_state_dict_hsdp_outer_optim(self):
        """HSDP outer-optim checkpoint state should survive a DCP roundtrip."""
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import Shard

        def build_model_and_optimizer(seed):
            torch.manual_seed(seed)
            model = SimpleMLP(64).to(device)
            fully_shard(
                model,
                mesh=mesh,
                sharding_strategy="optim_grads_params",
                outer_dp_sharding_strategy="optim",
                mp_policy=MixedPrecisionPolicy(
                    main_params_dtype=torch.float32, main_grads_dtype=torch.float32
                ),
                enable_async_reduce_grad=False,
            )
            assert all(
                isinstance(param_group, ParameterGroup)
                for param_group in model._fsdp_param_groups
            )
            return model, torch.optim.AdamW(model.parameters(), lr=1e-3)

        def run_one_step(model, optimizer, seed):
            torch.manual_seed(seed)
            x = torch.randn(2, 64, device=device)
            model.set_is_last_backward(True)
            loss = model(x).sum()
            loss.backward()
            model.finish_grad_sync()
            optimizer.step()

        def clone_dtensor_values(state_dict):
            return {
                name: value.to_local().detach().clone()
                for name, value in state_dict.items()
                if isinstance(value, DTensor)
            }

        def clone_optimizer_dtensor_values(state_dict):
            values = {}
            for name, state_tensors in state_dict.get("state", {}).items():
                values[name] = {
                    key: value.to_local().detach().clone()
                    for key, value in state_tensors.items()
                    if isinstance(value, DTensor) and value.to_local().dim() > 0
                }
            return {name: tensors for name, tensors in values.items() if tensors}

        def assert_hsdp_dtensor_metadata(dtensor):
            assert len(dtensor.placements) == 2
            assert isinstance(dtensor.placements[0], Shard)
            assert isinstance(dtensor.placements[1], Shard)
            assert hasattr(dtensor._local_tensor, "__create_chunk_list__")
            assert hasattr(dtensor._local_tensor, "__create_write_items__")

        device = _device()
        mesh = _build_hsdp_mesh()
        model, optimizer = build_model_and_optimizer(seed=42)
        run_one_step(model, optimizer, seed=43)

        model_sd, opt_sd = get_state_dict(model, optimizer)
        expected_model = clone_dtensor_values(model_sd)
        expected_optim = clone_optimizer_dtensor_values(opt_sd)
        assert expected_model, "HSDP model checkpoint should contain DTensor params"
        assert expected_optim, "HSDP optimizer checkpoint should contain DTensor state"

        for dtensor in (value for value in model_sd.values() if isinstance(value, DTensor)):
            assert_hsdp_dtensor_metadata(dtensor)
        for state_tensors in opt_sd.get("state", {}).values():
            for value in state_tensors.values():
                if isinstance(value, DTensor) and value.to_local().dim() > 0:
                    assert_hsdp_dtensor_metadata(value)

        ckpt_dir = Path(SHARED_TMP_DIR) / "test_get_state_dict_hsdp_outer_optim"
        if _rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()

        dcp.save({"model": model_sd, "optimizer": opt_sd}, checkpoint_id=str(ckpt_dir))
        torch.distributed.barrier()

        load_model, load_optimizer = build_model_and_optimizer(seed=123)
        run_one_step(load_model, load_optimizer, seed=124)
        load_model_sd, load_opt_sd = get_state_dict(load_model, load_optimizer)
        dcp.load({"model": load_model_sd, "optimizer": load_opt_sd}, checkpoint_id=str(ckpt_dir))
        torch_set_state_dict(
            load_model,
            load_optimizer,
            model_state_dict=load_model_sd,
            optim_state_dict=load_opt_sd,
            options=StateDictOptions(strict=False),
        )

        loaded_model_sd, loaded_opt_sd = get_state_dict(load_model, load_optimizer)
        loaded_model = clone_dtensor_values(loaded_model_sd)
        loaded_optim = clone_optimizer_dtensor_values(loaded_opt_sd)

        assert loaded_model.keys() == expected_model.keys()
        for name, expected in expected_model.items():
            assert torch.allclose(loaded_model[name], expected), name

        assert loaded_optim.keys() == expected_optim.keys()
        for name, expected_tensors in expected_optim.items():
            assert loaded_optim[name].keys() == expected_tensors.keys()
            for key, expected in expected_tensors.items():
                assert torch.allclose(loaded_optim[name][key], expected), f"{name}.{key}"

        if _rank() == 0:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        torch.distributed.barrier()
