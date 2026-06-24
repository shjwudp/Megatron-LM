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
modules. Forward and backward are captured as two separate ``CUDAGraph``
objects so that capture order matches runtime order when forward hooks fire
in module execution sequence.  The backward graph does NOT recompute the
forward — it reuses the autograd tape preserved during forward capture.

API
---
``FSDPCudaGraphRunner(fsdp_module, graph_pool=..., capture_stream=...)``
    * ``capture_forward(*args, **kwargs)`` — warmup + capture forward graph.
    * ``install()`` — patch ``module.forward`` to replay via autograd Function.
    * ``uninstall()`` — restore original ``forward``.

Backward capture is lazy: the first ``torch.autograd.grad`` triggers capture;
subsequent backwards replay the captured graph.
"""

import contextlib
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generator state helper
# ---------------------------------------------------------------------------


def _ensure_generator_graph_safe(device: Optional[int] = None) -> torch.Generator:
    """Fix inference-mode tensors in default generator state for CUDA graphs.

    Generator state tensors created under ``torch.inference_mode()`` cannot be
    updated in-place during CUDA graph capture.  Clone the state outside
    ``inference_mode`` and set it back so ``register_generator_state`` works.
    """
    if device is None:
        device = torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state = gen.get_state()
    if hasattr(state, "is_inference") and state.is_inference():
        with torch.inference_mode(mode=False):
            gen.set_state(state.clone())
    return gen


# ---------------------------------------------------------------------------
# Autograd function that replays CUDA graphs
# ---------------------------------------------------------------------------


class _CudaGraphFunction(torch.autograd.Function):
    """Custom autograd Function wrapping CUDA graph replay.

    ``forward`` replays ``runner.fwd_graph`` inside static-input buffers.
    ``backward`` lazily captures (first call) or replays ``runner.bwd_graph``.
    """

    @staticmethod
    def forward(ctx, runner, *flat_inputs):
        # Stage live inputs into graph-pool static buffers.
        for static, live in zip(runner.static_inputs, flat_inputs):
            if static.data_ptr() != live.data_ptr():
                static.copy_(live)

        runner.fwd_graph.replay()

        ctx.runner = runner
        ctx.save_for_backward(*flat_inputs)

        # Return detach() so downstream autograd does not alias static
        # outputs.  Replay order (all fwds then all bwds) guarantees
        # static_outputs are not overwritten before their backward runs.
        flat = tuple(o.detach() for o in runner.static_outputs)
        return runner._unflatten_output(flat)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        runner = ctx.runner

        # ---- first backward: capture ----
        if runner.bwd_graph is None:
            return runner._capture_backward(grad_outputs)

        # ---- subsequent backwards: replay ----
        return runner._replay_backward(grad_outputs)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class FSDPCudaGraphRunner:
    """Per-module split forward/backward CUDA graph runner.

    Forward is captured with grad enabled to preserve the autograd tape.
    Backward uses the tape directly — no forward recompute inside the
    backward graph.  ``saved_tensors_hooks`` clones saved tensors during
    forward capture so that fwd_graph replay does not bump their autograd
    version counters.

    Parameters
    ----------
    fsdp_module:
        The FSDP-wrapped module to graph.
    graph_pool:
        Shared ``torch.cuda.graph_pool_handle()``.  When multiple runners
        share a pool the CUDA driver reuses scratch memory across layers.
    capture_stream:
        Shared ``torch.cuda.Stream`` for graph capture.  All runners should
        use the same stream when sharing a pool so captures are serialised.
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

        # Saved during capture_forward.
        self.static_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self.static_outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._output_is_tuple: bool = False
        self._none_mask: Optional[List[bool]] = None
        self._tensor_param_names: List[str] = []
        self._frozen_kwargs: Dict[str, Any] = {}
        self._orig_forward = None

        # Backward replay state (created during backward capture).
        self._bwd_trainable_params: Optional[Tuple[torch.nn.Parameter, ...]] = None
        self._bwd_param_grad_bufs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_grad_outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._static_grad_inputs: Optional[Tuple[Optional[torch.Tensor], ...]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_forward(self, *args, **kwargs) -> None:
        """Warm up and capture the forward CUDA graph.

        Called by FSDP hooks during the first forward pass.  Saves
        the original ``module.forward`` before patching.
        """
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

        # ---- warmup (eager forward + backward) on throwaway stream ----
        #
        # Isolate warmup allocations (cuDNN benchmarking, lazy init) from both
        # the current stream and the capture stream so they don't pollute the
        # graph pool.
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

        # ---- record output structure (including None positions) ----
        out = self._run_forward(self.static_inputs)
        self._record_output_structure(out)
        self.static_outputs = self._flatten_output(out)

        # ---- capture forward graph (with grad enabled) ----
        #
        # Forward is captured WITH grad so the autograd tape stays alive
        # for backward capture (no forward recompute needed in bwd_graph).
        #
        # ``saved_tensors_hooks`` intercepts ``save_for_backward`` and saves
        # clones instead of the originals.  This prevents autograd version
        # mismatches when ``fwd_graph.replay()`` later modifies the originals
        # inplace (e.g., TE RoPE's ``freqs`` tensor).
        stream = self._capture_stream
        gen = _ensure_generator_graph_safe()
        self.fwd_graph = torch.cuda.CUDAGraph()
        self.fwd_graph.register_generator_state(gen)

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(
                self.fwd_graph, pool=self._graph_pool, stream=stream
            ):
                with torch.autograd.graph.saved_tensors_hooks(
                    lambda t: t.clone(), lambda t: t
                ):
                    self.static_outputs = self._flatten_output(
                        self._run_forward(self.static_inputs)
                    )

        stream.synchronize()
        self._captured = True
        logger.info("Forward graph captured for module id=%s", id(self._module))

    def install(self) -> None:
        """Replace ``module.forward`` with our autograd Function wrapper."""
        runner = self
        tensor_names = self._tensor_param_names

        def _cg_forward(*args, **kwargs):
            _, bound = runner._bind_forward_args(*args, **kwargs)
            bound.apply_defaults()
            flat = tuple(bound.arguments[n] for n in tensor_names)
            return _CudaGraphFunction.apply(runner, *flat)

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

        Uses ``self.static_outputs`` (which carry the autograd tape from
        forward capture) directly — no forward recompute.
        """
        stream = self._capture_stream
        flat_outputs = self.static_outputs

        # 1. Static grad-output buffers (clone live grads into graph pool).
        static_grad_outs = tuple(
            torch.zeros_like(o) if g is None else g.clone().detach()
            for o, g in zip(flat_outputs, grad_outputs)
        )

        # 2. Trainable params are not autograd.grad targets here.  We only
        #    initialize and remember their FSDP grad buffers so TE/Megatron
        #    fused backward can write main_grad as a side effect.
        trainable = tuple(p for p in self._module.parameters() if p.requires_grad)
        grad_bufs: Tuple[torch.Tensor, ...] = tuple(
            _get_param_grad_buffer(p) for p in trainable
        )
        self._bwd_trainable_params = trainable
        self._bwd_param_grad_bufs = grad_bufs

        # Clear param.grad so FSDP doesn't see stale grads.
        for p in self._module.parameters():
            if hasattr(p, "get_main_grad"):
                p.grad = None

        # 3. Autograd targets: static_inputs only.  Parameter gradients are
        #    handled by post-backward / TE side effects, not returned grads.
        input_targets = tuple(
            t for t in self.static_inputs if t.requires_grad
        )

        # 4. Capture bwd_graph: autograd.grad only (no forward recompute).
        torch.cuda.synchronize()

        gen = _ensure_generator_graph_safe()
        self.bwd_graph = torch.cuda.CUDAGraph()
        self.bwd_graph.register_generator_state(gen)

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
                        inputs=input_targets,
                        grad_outputs=grad_outs_for_capture,
                        retain_graph=False,
                        allow_unused=True,
                    )
                else:
                    grad_ins = ()

                n_input = len(input_targets)
                input_grads = grad_ins[:n_input]

            # Build static_grad_inputs aligned with self.static_inputs.
            grad_iter = iter(input_grads)
            self._static_grad_inputs = tuple(
                next(grad_iter) if t.requires_grad else None
                for t in self.static_inputs
            )
            self._static_grad_outputs = static_grad_outs

        # 5. Run the FIRST backward (using the just-captured graph).
        for s, l in zip(self._static_grad_outputs, grad_outputs):
            if l is not None and s.data_ptr() != l.data_ptr():
                s.copy_(l)
        _restore_param_grad_buffers(trainable, grad_bufs)
        self.bwd_graph.replay()

        logger.info("Backward graph captured for module id=%s", id(self._module))

        return (None,) + tuple(
            None if g is None else g.detach()
            for g in self._static_grad_inputs
        )

    def _replay_backward(
        self,
        grad_outputs: Tuple[Optional[torch.Tensor], ...],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Replay the previously captured backward graph."""
        for s, l in zip(self._static_grad_outputs, grad_outputs):
            if l is not None and s.data_ptr() != l.data_ptr():
                s.copy_(l)

        _restore_param_grad_buffers(
            self._bwd_trainable_params, self._bwd_param_grad_bufs
        )
        self.bwd_graph.replay()

        return (None,) + tuple(
            None if g is None else g.detach()
            for g in self._static_grad_inputs
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bind_forward_args(
        self, *args, **kwargs
    ) -> Tuple[inspect.Signature, inspect.BoundArguments]:
        """Bind args to the saved forward signature.

        ``self._orig_forward`` is normally a bound method, so its signature
        does not include ``self``.  Some wrappers may expose an unbound-style
        signature, so handle both forms explicitly.
        """
        sig = inspect.signature(self._orig_forward)
        if "self" in sig.parameters:
            return sig, sig.bind(self._module, *args, **kwargs)
        return sig, sig.bind(*args, **kwargs)

    def _run_forward(
        self, tensor_inputs: Tuple[torch.Tensor, ...]
    ) -> Any:
        """Run module.forward with the given tensor and frozen kwargs."""
        kw = dict(zip(self._tensor_param_names, tensor_inputs))
        kw.update(self._frozen_kwargs)
        return self._orig_forward(**kw)

    @staticmethod
    def _flatten_output(out: Any) -> Tuple[torch.Tensor, ...]:
        """Return a flat tuple of all non-``None`` tensors in *out*."""
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(t for t in out if isinstance(t, torch.Tensor))

    def _record_output_structure(self, out: Any) -> None:
        """Snapshot the output shape so ``_unflatten_output`` can restore it.

        Tracks whether the original output was a single tensor vs. a tuple,
        and which positions in the tuple were ``None``.
        """
        if isinstance(out, torch.Tensor):
            self._output_is_tuple = False
            self._none_mask = None
        elif isinstance(out, (tuple, list)):
            self._output_is_tuple = True
            self._none_mask = [t is None for t in out]
        else:
            self._output_is_tuple = False
            self._none_mask = None

    def _unflatten_output(
        self, flat: Tuple[torch.Tensor, ...]
    ) -> Any:
        """Restore the original output shape from a flat tensor tuple.

        Returns a single ``torch.Tensor`` when the original output was not a
        tuple, otherwise returns a tuple with ``None`` entries at their
        original positions.
        """
        if not self._output_is_tuple:
            return flat[0]
        if self._none_mask is None or not any(self._none_mask):
            return flat
        result = list(flat)
        for i, is_none in enumerate(self._none_mask):
            if is_none:
                result.insert(i, None)
        return tuple(result)


# ---------------------------------------------------------------------------
# Parameter gradient helpers (FSDP main_grad aware)
# ---------------------------------------------------------------------------


def _get_param_grad_buffer(param: torch.nn.Parameter) -> torch.Tensor:
    """Get or create a gradient buffer that FSDP's reduce-grad can consume."""
    if hasattr(param, "get_main_grad"):
        mg = param.get_main_grad()
        param.main_grad = mg
        return mg
    if param.grad is None:
        param.grad = torch.zeros_like(param)
    return param.grad


def _restore_param_grad_buffers(
    params: Tuple[torch.nn.Parameter, ...],
    buffers: Tuple[torch.Tensor, ...],
) -> None:
    """Restore param.main_grad / param.grad pointers before bwd replay."""
    for p, buf in zip(params, buffers):
        if hasattr(p, "get_main_grad"):
            p.main_grad = buf
            p.grad = None
        else:
            p.grad = buf
