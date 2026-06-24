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

"""CUDA graph capture / replay for individual FSDP modules.

Split forward/backward CUDA graph capture with a shared memory pool across
modules.  Mirrors ``torch.cuda.make_graphed_callables`` but with lazy
backward capture driven by autograd order instead of pre-scheduled reverse
capture.

Forward is captured with grad enabled so the autograd tape stays alive.
Backward uses the tape directly — no forward recompute.

Module parameters are passed through the autograd Function so their
gradients become visible to FSDP's post-backward hooks.

API
---
``FSDPCudaGraphRunner(fsdp_module, graph_pool=..., capture_stream=...)``
    * ``capture_forward(*args, **kwargs)`` — warmup + capture forward graph.
    * ``install()`` — patch ``module.forward`` to replay via autograd Function.
    * ``uninstall()`` — restore original ``forward``.
"""  # noqa: E501

import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    """Return True on rank 0, or True when not in a distributed context."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


# ---------------------------------------------------------------------------
# Generator state helper
# ---------------------------------------------------------------------------


def _ensure_generator_graph_safe(device: Optional[int] = None) -> torch.Generator:
    """Fix inference-mode tensors in default generator state for CUDA graphs."""
    if device is None:
        device = torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state = gen.get_state()
    if hasattr(state, "is_inference") and state.is_inference():
        with torch.inference_mode(mode=False):
            gen.set_state(state.clone())
    return gen


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------


class _CudaGraphFunction(torch.autograd.Function):
    """Custom autograd Function wrapping CUDA graph replay.

    ``forward`` receives ``(runner, *user_args, *module_params)`` so params
    participate in the autograd graph.  Only user args are staged into pool
    buffers — params are already at stable addresses.
    """

    @staticmethod
    def forward(ctx, runner, *flat_inputs):
        # flat_inputs = user_args + module_params
        n_user = runner._len_user_args

        # Stage live user args into static pool buffers.  Params stay as-is.
        for i in range(n_user):
            static = runner.static_inputs[i]
            live = flat_inputs[i]
            if static.data_ptr() != live.data_ptr():
                static.copy_(live)

        runner.fwd_graph.replay()
        ctx.runner = runner

        flat = tuple(o.detach() for o in runner.static_outputs)
        return runner._unflatten_output(flat)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        runner = ctx.runner

        if runner.bwd_graph is None:
            return runner._capture_backward(grad_outputs)

        return runner._replay_backward(grad_outputs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class FSDPCudaGraphRunner:
    """Per-module split forward/backward CUDA graph runner.

    Parameters
    ----------
    fsdp_module:
        The FSDP-wrapped module to graph.
    graph_pool:
        Shared ``torch.cuda.graph_pool_handle()``.
    capture_stream:
        Shared ``torch.cuda.Stream`` for serialised capture.
    num_warmup_iters:
        Number of eager forward+backward warmup passes before capture.
    """

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        graph_pool: Optional[Any] = None,
        capture_stream: Optional[torch.cuda.Stream] = None,
        num_warmup_iters: int = 3,
    ):
        self._module = fsdp_module
        self._graph_pool = graph_pool or torch.cuda.graph_pool_handle()
        self._capture_stream = capture_stream or torch.cuda.Stream()
        self._num_warmup = num_warmup_iters

        self.fwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self.bwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured: bool = False

        # Forward capture state.
        self.static_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self.static_outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._output_is_tuple: bool = False
        self._none_mask: Optional[List[bool]] = None
        self._tensor_param_names: List[str] = []
        self._frozen_kwargs: Dict[str, Any] = {}
        self._orig_forward = None

        # Backward replay state.
        self._len_user_args: int = 0
        self._module_params: Tuple[torch.nn.Parameter, ...] = ()
        self._static_grad_outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_grad_inputs: Optional[Tuple[Optional[torch.Tensor], ...]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_forward(self, *args, **kwargs) -> None:
        """Warm up and capture the forward CUDA graph."""
        if self._captured:
            return

        self._orig_forward = self._module.forward

        # ---- introspect forward signature ----
        sig, bound = self._bind_forward_args(*args, **kwargs)
        bound.apply_defaults()
        all_param_names = [n for n in sig.parameters if n != "self"]
        self._tensor_param_names = [
            n for n in all_param_names if isinstance(bound.arguments[n], torch.Tensor)
        ]
        self._frozen_kwargs = {
            n: bound.arguments[n]
            for n in all_param_names
            if n not in self._tensor_param_names
        }

        # ---- build static input buffers (inside the graph pool) ----
        live_inputs = tuple(bound.arguments[n] for n in self._tensor_param_names)
        self.static_inputs = tuple(
            t.clone().detach().requires_grad_(t.requires_grad) for t in live_inputs
        )
        self._len_user_args = len(self.static_inputs)
        self._module_params = tuple(self._module.parameters())

        # ---- warmup on throwaway stream ----
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(self._num_warmup):
                out = self._run_forward(self.static_inputs)
                flat = self._flatten_output(out)
                loss = sum(o.sum() for o in flat if o.requires_grad)
                if loss.requires_grad:
                    loss.backward()
                for p in self._module.parameters():
                    p.grad = None
        torch.cuda.synchronize()

        # ---- record output structure ----
        out = self._run_forward(self.static_inputs)
        self._record_output_structure(out)
        self.static_outputs = self._flatten_output(out)

        # ---- capture forward graph (with grad enabled) ----
        stream = self._capture_stream
        gen = _ensure_generator_graph_safe()
        self.fwd_graph = torch.cuda.CUDAGraph()
        self.fwd_graph.register_generator_state(gen)

        torch.cuda.reset_peak_memory_stats()
        _alloc_before = torch.cuda.memory_allocated()

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(
                self.fwd_graph, pool=self._graph_pool, stream=stream
            ):
                self.static_outputs = self._flatten_output(
                    self._run_forward(self.static_inputs)
                )

        stream.synchronize()
        self._captured = True

        _alloc_after = torch.cuda.memory_allocated()
        _peak_alloc = torch.cuda.max_memory_allocated()

        if _is_rank0():
            logger.info(
                "Forward graph captured for module id=%s: "
                "alloc %+.1f MB (%d→%d)  peak_alloc %d MB",
                id(self._module),
                (_alloc_after - _alloc_before) / 1e6,
                _alloc_before // 1_000_000,
                _alloc_after // 1_000_000,
                _peak_alloc // 1_000_000,
            )

    def install(self) -> None:
        """Replace ``module.forward`` with our autograd Function wrapper."""
        runner = self
        tensor_names = self._tensor_param_names
        params = self._module_params

        def _cg_forward(*args, **kwargs):
            _, bound = runner._bind_forward_args(*args, **kwargs)
            bound.apply_defaults()
            flat = tuple(bound.arguments[n] for n in tensor_names)
            return _CudaGraphFunction.apply(runner, *flat, *params)

        try:
            sig = inspect.signature(self._orig_forward)
            _cg_forward.__signature__ = sig
        except Exception:
            pass

        self._module.forward = _cg_forward

    def uninstall(self) -> None:
        """Restore the original ``module.forward``."""
        if self._orig_forward is not None:
            self._module.forward = self._orig_forward

    # ------------------------------------------------------------------
    # Backward capture / replay
    # ------------------------------------------------------------------

    def _capture_backward(
        self,
        grad_outputs: Tuple[Optional[torch.Tensor], ...],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Capture the backward graph on the first backward call.

        Mirrors ``make_graphed_callables``: ``torch.autograd.grad`` targets
        include both user args AND module params, so param gradients are
        captured into the graph and returned to autograd via ``backward()``.
        """
        stream = self._capture_stream
        flat_outputs = self.static_outputs

        # 1. Static grad-output buffers.
        static_grad_outs = tuple(
            torch.zeros_like(o) if g is None else g.clone().detach()
            for o, g in zip(flat_outputs, grad_outputs)
        )

        # 2. Build the full input surface: user args + module params.
        user_targets = tuple(t for t in self.static_inputs if t.requires_grad)
        param_targets = tuple(p for p in self._module_params if p.requires_grad)
        all_targets = user_targets + param_targets

        # 3. Capture bwd_graph.
        torch.cuda.synchronize()

        gen = _ensure_generator_graph_safe()
        self.bwd_graph = torch.cuda.CUDAGraph()
        self.bwd_graph.register_generator_state(gen)

        torch.cuda.reset_peak_memory_stats()
        _alloc_before = torch.cuda.memory_allocated()

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(
                self.bwd_graph, pool=self._graph_pool, stream=stream
            ):
                grad_outs_for_capture = tuple(
                    sg for sg, o in zip(static_grad_outs, flat_outputs)
                    if o.requires_grad
                )
                outputs_with_grad = tuple(
                    o for o in flat_outputs if o.requires_grad
                )

                if outputs_with_grad:
                    grad_ins = torch.autograd.grad(
                        outputs=outputs_with_grad,
                        inputs=all_targets,
                        grad_outputs=grad_outs_for_capture,
                        retain_graph=False,
                        allow_unused=True,
                    )
                else:
                    grad_ins = ()

            # 4. Build static_grad_inputs aligned with the FULL input surface
            #    (user args + params), with None for non-require-grad entries.
            grad_idx = 0
            full_surface = tuple(self.static_inputs) + self._module_params
            static_grad_inputs: List[Optional[torch.Tensor]] = []
            for t in full_surface:
                if isinstance(t, torch.Tensor) and t.requires_grad:
                    static_grad_inputs.append(grad_ins[grad_idx] if grad_ins else None)
                    grad_idx += 1
                else:
                    static_grad_inputs.append(None)

            self._static_grad_inputs = tuple(static_grad_inputs)
            self._static_grad_outputs = static_grad_outs

        _alloc_after = torch.cuda.memory_allocated()
        _peak_alloc = torch.cuda.max_memory_allocated()

        if _is_rank0():
            logger.info(
                "Backward graph captured for module id=%s: "
                "alloc %+.1f MB (%d→%d)  peak_alloc %d MB",
                id(self._module),
                (_alloc_after - _alloc_before) / 1e6,
                _alloc_before // 1_000_000,
                _alloc_after // 1_000_000,
                _peak_alloc // 1_000_000,
            )

        # 5. Run the FIRST backward.
        for s, l in zip(self._static_grad_outputs, grad_outputs):
            if l is not None and s.data_ptr() != l.data_ptr():
                s.copy_(l)
        self.bwd_graph.replay()

        # Return (None for runner, *user_grads, *param_grads).
        # Autograd will use param_grads to populate param.grad; FSDP
        # post-backward hooks then move them to main_grad.
        return (None,) + tuple(
            None if g is None else g.detach() for g in self._static_grad_inputs
        )

    def _replay_backward(
        self,
        grad_outputs: Tuple[Optional[torch.Tensor], ...],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Replay the previously captured backward graph."""
        for s, l in zip(self._static_grad_outputs, grad_outputs):
            if l is not None and s.data_ptr() != l.data_ptr():
                s.copy_(l)

        self.bwd_graph.replay()

        return (None,) + tuple(
            None if g is None else g.detach() for g in self._static_grad_inputs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bind_forward_args(
        self, *args, **kwargs
    ) -> Tuple[inspect.Signature, inspect.BoundArguments]:
        sig = inspect.signature(self._orig_forward)
        if "self" in sig.parameters:
            return sig, sig.bind(self._module, *args, **kwargs)
        return sig, sig.bind(*args, **kwargs)

    def _run_forward(
        self, tensor_inputs: Tuple[torch.Tensor, ...]
    ) -> Any:
        kw = dict(zip(self._tensor_param_names, tensor_inputs))
        kw.update(self._frozen_kwargs)
        return self._orig_forward(**kw)

    @staticmethod
    def _flatten_output(out: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(t for t in out if isinstance(t, torch.Tensor))

    def _record_output_structure(self, out: Any) -> None:
        if isinstance(out, torch.Tensor):
            self._output_is_tuple = False
            self._none_mask = None
        elif isinstance(out, (tuple, list)):
            self._output_is_tuple = True
            self._none_mask = [t is None for t in out]
        else:
            self._output_is_tuple = False
            self._none_mask = None

    def _unflatten_output(self, flat: Tuple[torch.Tensor, ...]) -> Any:
        if not self._output_is_tuple:
            return flat[0]
        if self._none_mask is None or not any(self._none_mask):
            return flat
        result = list(flat)
        for i, is_none in enumerate(self._none_mask):
            if is_none:
                result.insert(i, None)
        return tuple(result)
