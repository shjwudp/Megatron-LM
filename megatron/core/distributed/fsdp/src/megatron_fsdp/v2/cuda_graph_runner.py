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

"""Single-pass CUDA graph capture for FSDP modules with shared-pool and
shared-buffer memory reuse.

Memory optimization: shared static buffers
==========================================

In a transformer model with N identical layers, every forward graph reads
from ``static_inputs`` (shape [B, S, H]) and every backward graph reads
from ``static_grad_outputs`` (same shape). These are used SEQUENTIALLY
(fwd0 -> fwd1 -> ... -> fwdN; bwdN -> ... -> bwd1 -> bwd0), never
concurrently.

By capturing ALL forward graphs reading from the SAME buffer address, and
ALL backward graphs reading from the SAME grad buffer address, we reduce:

    N x 2 x B*S*H  ->  1 x 2 x B*S*H   (for the input/grad buffers)

For 80 layers, B=1, S=8192, H=12288, bf16: ~30 GB -> 384 MB.

The ``CudaGraphPool`` manages shared buffers keyed by (shape, dtype).
Each runner captures its graph reading from the shared buffer. At replay,
we ``copy_`` the live tensor into the shared buffer then ``replay()``.

Lifecycle (one pass, driven by FSDP hooks)
==========================================

::

    First microbatch (capture):
      forward_pre_hook(layer_i):
        1. Acquire shared input buffer from pool (by shape)
        2. Pop hooks, warmup, capture fwd graph reading from shared buffer
        3. Restore hooks, return output
      backward_pre_hook(layer_i):
        1. Acquire shared grad-output buffer from pool (by shape)
        2. Pop hooks, capture bwd graph reading from shared grad buffer
        3. Restore hooks, release fwd output memory hint
        4. Replay bwd graph with live grads, return grad_inputs
      post_backward_final:
        install_all()

    Second microbatch onward (replay):
      _CudaGraphFunction.forward:
        copy input -> shared buffer, replay fwd graph, clone outputs
      _CudaGraphFunction.backward:
        copy grads -> shared buffer, replay bwd graph, clone grad_inputs
"""

import gc
import inspect
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


_CG_NO_GRAD_FWD: bool = os.environ.get(
    "MFSDP_CG_NO_GRAD_FWD", os.environ.get("_CG_NO_GRAD_FWD", "0")
).lower() in ("1", "true", "yes", "on")
_CG_MEM_DEBUG: bool = os.environ.get("MFSDP_CG_MEM_DEBUG", "0").lower() in (
    "1",
    "cg",
    "true",
    "yes",
    "on",
)


# ------------------------------------------------------------------
# Tensor aliasing trick (inspired by TE's _WeakRefTensor / make_weak_ref)
# ------------------------------------------------------------------


def _make_viewless_tensor(src: torch.Tensor, requires_grad: bool) -> torch.Tensor:
    """Create a NEW tensor that shares memory with *src* but is a fresh
    leaf in autograd -- no ._base, no grad_fn, independently requires_grad.

    This bypasses the "leaf Variable used in in-place operation" error
    because the returned tensor is a brand-new leaf that happens to
    alias the same storage. PyTorch only checks the Python tensor
    object's autograd metadata, not the raw data_ptr.

    Equivalent to TE's `safely_set_viewless_tensor_data` pattern.
    """
    new_tensor = torch.empty([], dtype=src.dtype, device=src.device)
    new_tensor.set_(
        src.untyped_storage(),
        storage_offset=src.storage_offset(),
        size=src.shape,
        stride=src.stride(),
    )
    new_tensor.requires_grad_(requires_grad)
    return new_tensor


# ------------------------------------------------------------------
# Hook helpers
# ------------------------------------------------------------------

_HOOK_ATTRS = [
    "_forward_pre_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]


def _get_forward_param_names(module_cls) -> List[str]:
    sig = inspect.signature(module_cls.forward)
    return [
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]


def _pop_hooks_recursive(
    module: torch.nn.Module,
) -> List[Tuple[torch.nn.Module, Dict[str, Any]]]:
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []
    for submodule in module.modules():
        snap: Dict[str, Any] = {}
        for attr in _HOOK_ATTRS:
            if hasattr(submodule, attr):
                snap[attr] = getattr(submodule, attr)
                setattr(submodule, attr, OrderedDict())
        saved.append((submodule, snap))
    return saved


def _restore_hooks_recursive(
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]],
) -> None:
    for submodule, snap in saved:
        for name, value in snap.items():
            if value is not None:
                setattr(submodule, name, value)


def _ensure_generator_graph_safe(device: Optional[int] = None):
    if device is None:
        device = torch.cuda.current_device()
    gen = torch.cuda.default_generators[device]
    state = gen.get_state()
    if hasattr(state, "is_inference") and state.is_inference():
        with torch.no_grad():
            gen.set_state(state.clone())
    return gen


# ------------------------------------------------------------------
# Shared pool + buffer manager
# ------------------------------------------------------------------


class CudaGraphPool:
    """Shared CUDA graph memory pool and reusable static buffer registry.

    Manages:
    * A shared ``graph_pool_handle`` for all captured graphs (workspace reuse).
    * A registry of shared static buffers keyed by (shape, dtype, direction).
      Multiple runners with the same input shape share ONE buffer.
    * A single capture stream for ordered capture.

    Parameters
    ----------
    pool:
        Optional existing pool handle. If None, creates a new one.
    """

    def __init__(self, pool: Optional[Any] = None):
        self.pool: Any = pool or torch.cuda.graph_pool_handle()
        self.capture_stream: torch.cuda.Stream = torch.cuda.Stream()

        # Shared buffer registry: (shape_tuple, dtype, direction) -> Tensor
        # direction is "input" or "grad_output"
        self._shared_buffers: Dict[Tuple[Tuple[int, ...], torch.dtype, str], torch.Tensor] = {}

    def get_shared_input_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
        requires_grad: bool = True,
    ) -> torch.Tensor:
        """Get or create a shared static input buffer for forward graphs.

        All layers with the same input shape/dtype share ONE buffer.
        Each fwd graph is captured reading from this address; at replay,
        we copy_ into it before replay().
        """
        key = (tuple(shape), dtype, "input")
        if key not in self._shared_buffers:
            buf = torch.empty(
                shape, dtype=dtype, device=device
            ).requires_grad_(requires_grad)
            self._shared_buffers[key] = buf
            logger.debug(
                "CudaGraphPool: allocated shared input buffer %s %s (%.1f MB)",
                shape, dtype, buf.nelement() * buf.element_size() / 1e6,
            )
        return self._shared_buffers[key]

    def get_shared_grad_output_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Get or create a shared static grad-output buffer for bwd graphs.

        All layers with the same grad shape/dtype share ONE buffer.
        Each bwd graph is captured reading from this address.
        """
        key = (tuple(shape), dtype, "grad_output")
        if key not in self._shared_buffers:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._shared_buffers[key] = buf
            logger.debug(
                "CudaGraphPool: allocated shared grad_output buffer %s %s (%.1f MB)",
                shape, dtype, buf.nelement() * buf.element_size() / 1e6,
            )
        return self._shared_buffers[key]

    def get_shared_buffers_for_inputs(
        self,
        sample_tensors: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Get shared input buffers matching a tuple of sample tensors.

        Each position gets its own shared buffer (keyed by shape+dtype+index
        to handle multiple input tensors with different shapes).

        NOTE: The returned buffers do NOT have requires_grad set. Callers
        must use _make_viewless_tensor() to create a grad-enabled alias
        for autograd without triggering in-place errors on copy_().
        """
        buffers = []
        for i, t in enumerate(sample_tensors):
            key = (tuple(t.shape), t.dtype, f"input_{i}")
            if key not in self._shared_buffers:
                # Allocate WITHOUT requires_grad -- it's just raw storage.
                buf = torch.empty(t.shape, dtype=t.dtype, device=t.device)
                self._shared_buffers[key] = buf
            buffers.append(self._shared_buffers[key])
        return tuple(buffers)

    def get_shared_buffers_for_grad_outputs(
        self,
        sample_tensors: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Get shared grad-output buffers matching a tuple of output tensors.

        Like input buffers, these are raw (no requires_grad) storage.
        """
        buffers = []
        for i, t in enumerate(sample_tensors):
            key = (tuple(t.shape), t.dtype, f"grad_output_{i}")
            if key not in self._shared_buffers:
                buf = torch.empty(t.shape, dtype=t.dtype, device=t.device)
                self._shared_buffers[key] = buf
            buffers.append(self._shared_buffers[key])
        return tuple(buffers)

    def synchronize(self):
        torch.cuda.current_stream().wait_stream(self.capture_stream)
        torch.cuda.synchronize()

    def reset(self):
        self._shared_buffers.clear()


# ------------------------------------------------------------------
# Autograd Function for replay
# ------------------------------------------------------------------


class _CudaGraphFunction(torch.autograd.Function):
    """Replay fwd graph on forward, bwd graph on backward.

    Static buffers are SHARED across layers -- the copy_ into them is
    what "selects" which layer's data is active.
    """

    @staticmethod
    def forward(ctx, runner, *flat_inputs):
        # Copy live inputs -> raw (no-grad) shared buffers.
        # The fwd graph reads from the same addresses (via viewless alias).
        for raw_buf, live in zip(runner._raw_input_buffers, flat_inputs):
            if raw_buf.data_ptr() != live.data_ptr():
                raw_buf.copy_(live)

        # Replay forward graph
        runner._fwd_graph.replay()

        ctx.runner = runner
        if _CG_NO_GRAD_FWD:
            ctx.save_for_backward(*flat_inputs)

        # Clone outputs OUT of pool immediately (before next layer's
        # fwd replay could overwrite shared pool memory).
        cloned = tuple(o.clone() for o in runner._static_outputs)
        return cloned

    @staticmethod
    def backward(ctx, *grad_outputs):
        runner = ctx.runner

        # First backward: capture bwd graph lazily
        if runner._bwd_graph is None:
            result = runner.capture_backward(grad_outputs, ctx.saved_tensors)
            return (None,) + result

        # Subsequent backwards: replay
        if runner._bwd_inputs:
            with torch.no_grad():
                for static, live in zip(runner._bwd_inputs, ctx.saved_tensors):
                    if static.data_ptr() != live.data_ptr():
                        static.copy_(live)

        for raw_buf, live in zip(runner._raw_grad_output_buffers, grad_outputs):
            if live is None:
                continue
            if raw_buf.data_ptr() != live.data_ptr():
                raw_buf.copy_(live)

        # Replay backward graph
        runner._bwd_graph.replay()

        # (None for runner) + cloned grad_inputs
        cloned = tuple(
            None if g is None else g.clone()
            for g in runner._static_grad_inputs
        )
        return (None,) + cloned


# ------------------------------------------------------------------
# Per-module runner
# ------------------------------------------------------------------


class FSDPCudaGraphRunner:
    """Per-module CUDA graph runner with shared static buffers.

    Captures forward and backward graphs inline (one pass) during the
    first microbatch. Uses shared input/grad buffers from ``CudaGraphPool``
    so N layers with the same shape share a single buffer (~30GB savings
    for large models).

    Parameters
    ----------
    fsdp_module:
        The FSDP module to graph.
    shared_pool:
        ``CudaGraphPool`` instance shared across all modules.
    num_warmup_iters:
        Warmup iterations before capture (default 3).
    """

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        shared_pool: CudaGraphPool,
        num_warmup_iters: int = 3,
    ):
        self._module = fsdp_module
        self._pool = shared_pool
        self._num_warmup_iters = num_warmup_iters

        # Module identifier for debug logging
        self._module_name = getattr(fsdp_module, "_fsdp_module_name", fsdp_module.__class__.__name__)
        self._log_prefix = f"[{self._module_name}]"

        # Forward state (static_inputs is SHARED across layers)
        self._fwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_inputs: Tuple[torch.Tensor, ...] = ()
        self._raw_input_buffers: Tuple[torch.Tensor, ...] = ()
        self._static_outputs: Tuple[torch.Tensor, ...] = ()

        # Backward state (static_grad_outputs is SHARED across layers)
        self._bwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_grad_outputs: Tuple[torch.Tensor, ...] = ()
        self._raw_grad_output_buffers: Tuple[torch.Tensor, ...] = ()
        self._static_grad_inputs: Tuple[Optional[torch.Tensor], ...] = ()
        self._bwd_inputs: Tuple[torch.Tensor, ...] = ()

        # Metadata
        self._tensor_param_names: List[str] = []
        self._param_names: List[str] = []
        self._frozen_kwargs: Dict[str, Any] = {}
        self._output_is_tuple: bool = True
        self._none_mask: Optional[List[bool]] = None

        # State
        self._fwd_captured: bool = False
        self._bwd_captured: bool = False
        self._installed: bool = False
        self._orig_fwd: Optional[Any] = None

    # ------------------------------------------------------------------
    # Forward capture
    # ------------------------------------------------------------------

    def capture_forward(self, *sample_args, **sample_kwargs) -> Any:
        """Capture forward graph inline. Returns the module's output.

        The forward graph reads from SHARED static input buffers
        (acquired from CudaGraphPool). All layers with same input shape
        read from the same address.
        """
        # ---- 1. Parse inputs ----
        param_names = _get_forward_param_names(self._module.__class__)
        bound: Dict[str, Any] = {}
        for i, val in enumerate(sample_args):
            if i < len(param_names):
                bound[param_names[i]] = val
        bound.update(sample_kwargs)

        tensor_names = [
            n for n in param_names
            if n in bound and isinstance(bound[n], torch.Tensor)
        ]
        frozen_kwargs = {
            n: v for n, v in bound.items()
            if n not in tensor_names
        }

        # ---- 2. Acquire SHARED static input buffers ----
        sample_tensors = tuple(bound[n] for n in tensor_names)
        raw_buffers = self._pool.get_shared_buffers_for_inputs(sample_tensors)

        # Copy sample data into shared buffers.
        # raw_buffers have NO requires_grad -> copy_ is safe (no autograd error).
        for buf, sample in zip(raw_buffers, sample_tensors):
            buf.copy_(sample.detach())

        # Create viewless aliases WITH requires_grad for graph capture.
        # These are fresh leaves sharing the same memory -- autograd sees
        # them as new tensors, so future copy_() on the raw buffer won't
        # trigger "leaf Variable used in in-place operation".
        static_inputs = tuple(
            _make_viewless_tensor(buf, requires_grad=sample.requires_grad)
            for buf, sample in zip(raw_buffers, sample_tensors)
        )

        self._static_inputs = static_inputs
        # Keep raw buffer refs for replay-time copy_ (no grad -> no error)
        self._raw_input_buffers = raw_buffers
        self._tensor_param_names = tensor_names
        self._param_names = param_names
        self._frozen_kwargs = frozen_kwargs

        # Zero grads, unshard main grad buffer
        for param in self._module.parameters():
            param.grad = None

        # Pop hooks
        saved_hooks = _pop_hooks_recursive(self._module)

        gc.collect()

        try:
            stream = self._pool.capture_stream
            torch.cuda.synchronize()

            # ---- 3. Warmup ----
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(self._num_warmup_iters):
                    out = self._call_module(static_inputs)
                    flat_out = self._flatten_output(out)
                    if any(o.requires_grad for o in flat_out):
                        torch.autograd.grad(
                            outputs=tuple(o for o in flat_out if o.requires_grad),
                            inputs=tuple(
                                t for t in static_inputs + tuple(self._module.parameters())
                                if t.requires_grad
                            ),
                            grad_outputs=tuple(
                                torch.empty_like(o) for o in flat_out if o.requires_grad
                            ),
                            only_inputs=True,
                            allow_unused=True,
                        )
                    del out, flat_out
                for param in self._module.parameters():
                    param.grad = None
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()
            del warmup_stream

            # Full cleanup: collect cyclic autograd garbage from warmup, THEN
            # freeze GC and release freed blocks back to CUDA.
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()  # second pass catches ref-cycles broken by first collect
            torch.cuda.empty_cache()

            # ---- 4. Capture forward ----
            gen = _ensure_generator_graph_safe()
            self._fwd_graph = torch.cuda.CUDAGraph()
            self._fwd_graph.register_generator_state(gen)

            # ---- INSTRUMENTATION: Track memory during capture ----
            if _CG_MEM_DEBUG:
                torch.cuda.memory._record_memory_history(
                    enabled='all', context='all', stacks='python', max_entries=500000
                )
            torch.cuda.reset_peak_memory_stats()
            _before_reserved = torch.cuda.memory_reserved()
            _before_allocated = torch.cuda.memory_allocated()

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(self._fwd_graph, pool=self._pool.pool, stream=stream):
                    if _CG_NO_GRAD_FWD:
                        with torch.no_grad():
                            out = self._call_module(static_inputs)
                    else:
                        out = self._call_module(static_inputs)

            torch.cuda.synchronize()
            _after_reserved = torch.cuda.memory_reserved()
            _after_allocated = torch.cuda.memory_allocated()
            _peak_reserved = torch.cuda.max_memory_reserved()
            _peak_allocated = torch.cuda.max_memory_allocated()

            logger.info(
                "%s CAPTURE MEMORY: reserved_delta=%.1f MB, allocated_delta=%.1f MB, "
                "peak_reserved=%.1f MB, peak_allocated=%.1f MB, "
                "peak_reserved_delta=%.1f MB, peak_allocated_delta=%.1f MB",
                self._log_prefix,
                (_after_reserved - _before_reserved) / 1e6,
                (_after_allocated - _before_allocated) / 1e6,
                _peak_reserved / 1e6,
                _peak_allocated / 1e6,
                (_peak_reserved - _before_reserved) / 1e6,
                (_peak_allocated - _before_allocated) / 1e6,
            )

            # Dump full snapshot for the FIRST captured layer only
            if _CG_MEM_DEBUG and not hasattr(self._pool, '_first_layer_dumped'):
                self._pool._first_layer_dumped = True
                torch.cuda.memory._dump_snapshot('/tmp/first_layer_capture.pickle')
                logger.info(
                    "%s Dumped memory snapshot to /tmp/first_layer_capture.pickle "
                    "(visualize at https://pytorch.org/memory_viz)",
                    self._log_prefix,
                )

            if _CG_MEM_DEBUG:
                torch.cuda.memory._record_memory_history(enabled=None)
            # ---- END INSTRUMENTATION ----

            self._record_output_structure(out)
            self._static_outputs = tuple(self._flatten_output(out))
            self._fwd_captured = True

        finally:
            _restore_hooks_recursive(saved_hooks)

        return self._call_module(static_inputs)

    # ------------------------------------------------------------------
    # Backward capture
    # ------------------------------------------------------------------

    def capture_backward(
        self,
        grad_outputs: Tuple[torch.Tensor, ...],
        saved_inputs: Tuple[torch.Tensor, ...] = (),
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Capture backward graph inline. Returns grad_inputs.

        The backward graph reads from SHARED static grad-output buffers
        (acquired from CudaGraphPool). All layers with same output shape
        read from the same grad address.
        """
        assert self._fwd_captured, "Must capture forward first"

        # Pop hooks for clean capture
        saved_hooks = _pop_hooks_recursive(self._module)

        try:
            stream = self._pool.capture_stream
            torch.cuda.synchronize()

            # ---- 1. Acquire SHARED static grad-output buffers ----
            raw_grad_buffers = self._pool.get_shared_buffers_for_grad_outputs(
                self._static_outputs
            )

            # Copy live grads into raw buffers (no requires_grad -> safe)
            for buf, live in zip(raw_grad_buffers, grad_outputs):
                if live is not None:
                    buf.copy_(live.detach())

            # Viewless aliases for capture (fresh leaves, same memory)
            static_grad_outputs = tuple(
                _make_viewless_tensor(buf, requires_grad=False)
                for buf in raw_grad_buffers
            )

            self._static_grad_outputs = static_grad_outputs
            self._raw_grad_output_buffers = raw_grad_buffers

            # Zero grads
            for param in self._module.parameters():
                param.grad = None
            self._unshard_main_grad_buffer()

            # ---- 2. Capture backward ----
            gen = _ensure_generator_graph_safe()

            if _CG_NO_GRAD_FWD:
                recompute_inputs = tuple(
                    t.detach().clone().requires_grad_(t.requires_grad)
                    for t in saved_inputs
                )
                self._bwd_inputs = recompute_inputs
                input_tensors = recompute_inputs
                flat_outputs = None
            else:
                self._bwd_inputs = ()
                input_tensors = self._static_inputs
                flat_outputs = self._static_outputs

            inputs_for_capture = tuple(t for t in input_tensors if t.requires_grad)

            self._bwd_graph = torch.cuda.CUDAGraph()
            self._bwd_graph.register_generator_state(gen)

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(self._bwd_graph, pool=self._pool.pool, stream=stream):
                    if _CG_NO_GRAD_FWD:
                        recompute_out = self._call_module(input_tensors)
                        flat_outputs = self._flatten_output(recompute_out)

                    assert flat_outputs is not None
                    outputs_for_capture = tuple(o for o in flat_outputs if o.requires_grad)
                    grad_outputs_for_capture = tuple(
                        sg for sg, o in zip(static_grad_outputs, flat_outputs)
                        if o.requires_grad
                    )

                    grad_ins = torch.autograd.grad(
                        outputs=outputs_for_capture,
                        inputs=inputs_for_capture,
                        grad_outputs=grad_outputs_for_capture,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                        allow_unused=True,
                    )

            # Map back to input positions
            grad_iter = iter(grad_ins)
            static_grad_inputs: List[Optional[torch.Tensor]] = []
            for t in input_tensors:
                if t.requires_grad:
                    static_grad_inputs.append(next(grad_iter))
                else:
                    static_grad_inputs.append(None)

            self._static_grad_inputs = tuple(static_grad_inputs)
            self._bwd_captured = True

        finally:
            _restore_hooks_recursive(saved_hooks)
            self._reshard_main_grad_buffer()

        # Run first real backward with live grads
        if self._bwd_inputs:
            with torch.no_grad():
                for static, live in zip(self._bwd_inputs, saved_inputs):
                    if static.data_ptr() != live.data_ptr():
                        static.copy_(live)

        for raw_buf, live in zip(self._raw_grad_output_buffers, grad_outputs):
            if live is None:
                continue
            raw_buf.copy_(live)
        self._bwd_graph.replay()

        result = tuple(
            None if g is None else g.clone()
            for g in self._static_grad_inputs
        )
        return result

    # ------------------------------------------------------------------
    # Install / uninstall
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Patch module.forward -> replay graphs via _CudaGraphFunction."""
        if not self._fwd_captured:
            raise RuntimeError("Forward must be captured before install")
        if self._installed:
            return

        self._orig_fwd = self._module.forward
        runner = self
        param_names = self._param_names
        tensor_names = self._tensor_param_names

        def _patched_fwd(*args, **kwargs):
            bound: Dict[str, Any] = {}
            for i, val in enumerate(args):
                if i < len(param_names):
                    bound[param_names[i]] = val
            bound.update(kwargs)
            flat = tuple(bound[n] for n in tensor_names)
            outs = _CudaGraphFunction.apply(runner, *flat)
            return runner._unflatten_output(outs)

        self._module.forward = _patched_fwd
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._module.forward = self._orig_fwd
        self._orig_fwd = None
        self._installed = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_module(self, flat_inputs: Tuple[torch.Tensor, ...]) -> Any:
        kwargs = dict(zip(self._tensor_param_names, flat_inputs))
        kwargs.update(self._frozen_kwargs)
        forward = self._orig_fwd if self._orig_fwd is not None else self._module.forward
        return forward(**kwargs)

    def _record_output_structure(self, out: Any) -> None:
        if isinstance(out, torch.Tensor):
            self._output_is_tuple = False
            self._none_mask = None
        elif isinstance(out, (tuple, list)):
            self._output_is_tuple = True
            self._none_mask = [t is None for t in out]
        else:
            raise RuntimeError(f"Unsupported output type: {type(out)}")

    def _flatten_output(self, out: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(t for t in out if isinstance(t, torch.Tensor))

    def _unflatten_output(self, flat: Tuple[torch.Tensor, ...]) -> Any:
        if not self._output_is_tuple:
            return flat[0]
        if self._none_mask is None:
            return flat
        full: List[Any] = []
        it = iter(flat)
        for is_none in self._none_mask:
            full.append(None if is_none else next(it))
        return tuple(full)

    def _unshard_main_grad_buffer(self) -> None:
        for group in getattr(self._module, "_fsdp_param_groups", []):
            if hasattr(group, "main_grad_buffer") and group.main_grad_buffer is not None:
                group.main_grad_buffer.fetch_buffer()

    def _reshard_main_grad_buffer(self) -> None:
        for group in getattr(self._module, "_fsdp_param_groups", []):
            if hasattr(group, "release_grad_buffer"):
                group.release_grad_buffer()

    @property
    def captured(self) -> bool:
        return self._fwd_captured and self._bwd_captured

    @property
    def using_cuda_graph(self) -> bool:
        return self._installed

    def reset(self) -> None:
        self.uninstall()
        self._fwd_graph = None
        self._bwd_graph = None
        self._static_inputs = ()
        self._static_outputs = ()
        self._static_grad_outputs = ()
        self._static_grad_inputs = ()
        self._bwd_inputs = ()
        self._fwd_captured = False
        self._bwd_captured = False
