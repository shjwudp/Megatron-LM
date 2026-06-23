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
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .inspect_autograd_tape import inspect_autograd_tape

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Non-owning tensor wrapper — drops storage ownership so the CUDA
# allocator can reuse the block while keeping a typed view at the
# same data_ptr.  On replay the graph writes to the same address,
# so the weakref tensor sees fresh data with no extra copies.
# ------------------------------------------------------------------

_TENSOR_TYPE_TO_NP = {
    torch.float16: "<f2",
    torch.float32: "<f4",
    torch.int64: "<i8",
    torch.int32: "<i4",
    torch.int8: "|i1",
    torch.bool: "|b1",
    torch.bfloat16: "<f2",
}


class _WeakRefTensor:
    def __init__(self, data_ptr: int, dtype: torch.dtype, shape):
        self._data_ptr = data_ptr
        self.dtype = dtype
        self.shape = shape

    def data_ptr(self):
        return self._data_ptr

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def __cuda_array_interface__(self):
        n = 1
        for d in self.shape:
            n *= int(d)
        return {
            "shape": self.shape,
            "typestr": _TENSOR_TYPE_TO_NP[self.dtype],
            "data": (self._data_ptr if n > 0 else 0, False),
            "version": 3,
        }


def make_weak_ref(x):
    if isinstance(x, torch.Tensor):
        if not x.is_cuda:
            return x
        weak = _WeakRefTensor(x.data_ptr(), x.dtype, x.shape)
        # torch.as_tensor on __cuda_array_interface__ gives non-owning storage.
        return torch.as_tensor(weak).view(x.dtype)
    if x is None:
        return x
    if isinstance(x, tuple):
        return tuple(make_weak_ref(i) for i in x)
    raise TypeError(f"make_weak_ref: unsupported type {type(x).__name__}")


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

        # Clone outputs OUT of pool immediately (before next layer's
        # fwd replay could overwrite shared pool memory).
        cloned = tuple(o.clone() for o in runner._static_outputs)
        return cloned

    @staticmethod
    def backward(ctx, *grad_outputs):
        runner = ctx.runner

        # First backward: capture bwd graph lazily
        if runner._bwd_graph is None:
            result = runner.capture_backward(grad_outputs)
            return (None,) + result

        # Subsequent backwards: replay
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
    # Tape-reuse experiment (CG vs no-CG)
    # ------------------------------------------------------------------

    _tape_reuse_experiment: bool = False
    _tape_keep_experiment: bool = False
    _tape_experiment_max_layers: int = 13
    _tape_ptr_history: List[List[Tuple[int, Tuple[int, ...], int]]] = []
    _tape_cg_history: List[List[Tuple[int, Tuple[int, ...], int]]] = []
    _tape_nocg_keep_history: List[List[Tuple[int, Tuple[int, ...], int]]] = []
    _kept_tape_tensors_all: List[torch.Tensor] = []

    @classmethod
    def enable_tape_reuse_experiment(cls, max_layers: int = 13):
        """No-CG, _pack returns data_ptr (drops ref → allocator recycles)."""
        cls._tape_reuse_experiment = True
        cls._tape_keep_experiment = False
        cls._tape_experiment_max_layers = max_layers
        cls._tape_ptr_history.clear()
        cls._tape_cg_history.clear()
        cls._tape_nocg_keep_history.clear()
        cls._kept_tape_tensors_all.clear()

    @classmethod
    def enable_tape_keep_experiment(cls, max_layers: int = 13):
        """No-CG, _pack returns tensor (keeps ref, like CG).

        Only holds refs for the first ``max_layers`` layers to avoid OOM.
        After that, continues tracking data_ptrs but doesn't hold refs —
        those layers' tensors get freed, allowing address recycling.

        The first N layers show "all tapes alive" behavior (fair CG baseline).
        """
        cls._tape_reuse_experiment = False
        cls._tape_keep_experiment = True
        cls._tape_experiment_max_layers = max_layers
        cls._tape_ptr_history.clear()
        cls._tape_cg_history.clear()
        cls._tape_nocg_keep_history.clear()
        cls._kept_tape_tensors_all.clear()

    @classmethod
    def disable_tape_reuse_experiment(cls):
        cls._tape_reuse_experiment = False
        cls._tape_keep_experiment = False

    @classmethod
    def print_tape_reuse_report(cls):
        cls._print_tape_report(cls._tape_ptr_history, "TAPE-EXPERIMENT (no-CG, free-ref)")

    @classmethod
    def print_tape_keep_report(cls):
        cls._print_tape_report(cls._tape_nocg_keep_history, "TAPE-KEEP (no-CG, keep-ref)")

    @classmethod
    def print_tape_cg_report(cls):
        cls._print_tape_report(cls._tape_cg_history, "TAPE-CG")

    @classmethod
    def _print_tape_report(cls, history, label):
        if not history:
            logger.info("%s: no data collected.", label)
            return

        # Per data_ptr: which layers used it, and shape/size info (from first occurrence)
        ptr_to_layers: Dict[int, List[int]] = {}
        ptr_info: Dict[int, Tuple[Tuple[int, ...], int]] = {}
        for idx, entries in enumerate(history):
            for ptr, shape, size in entries:
                ptr_to_layers.setdefault(ptr, []).append(idx)
                if ptr not in ptr_info:
                    ptr_info[ptr] = (shape, size)

        reused = sum(1 for layers in ptr_to_layers.values() if len(layers) > 1)
        total = len(ptr_to_layers)
        max_reuse = max((len(v) for v in ptr_to_layers.values()), default=0)

        # Aggregate by shape: total size, how many unique ptrs of this shape, max reuse
        shape_stats: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        for ptr, layers in ptr_to_layers.items():
            shape, size = ptr_info[ptr]
            if shape not in shape_stats:
                shape_stats[shape] = {"size": size, "count": 0, "max_reuse_idx": 0, "total_layers": 0}
            shape_stats[shape]["count"] += 1
            shape_stats[shape]["max_reuse_idx"] = max(shape_stats[shape]["max_reuse_idx"], len(layers))
            shape_stats[shape]["total_layers"] += len(layers)

        # Saved memory: if all ptrs of a shape were reused across N layers,
        # the saving = (total_layers_using - unique_ptrs) × size
        saved_mb = 0.0
        layers = len(history)
        for shape, stats in shape_stats.items():
            extra = stats["total_layers"] - stats["count"]
            saved_mb += extra * stats["size"] / 1e6

        logger.info(
            "%s: %d layers, %d unique data_ptrs, "
            "%d reused across ≥2 layers (%.1f%%), max same-addr layers=%d. "
            "Saved by reuse: %.1f MB",
            label, layers, total, reused,
            100 * reused / total if total else 0, max_reuse,
            saved_mb,
        )
        for shape, stats in sorted(shape_stats.items(), key=lambda x: -x[1]["size"] * max(0, x[1]["total_layers"] - x[1]["count"]))[:5]:
            sz = stats["size"] / 1e6
            extra = stats["total_layers"] - stats["count"]
            if extra > 0:
                logger.info(
                    "  shape=%s  %.1f MB×%d reuse  →  saved %.1f MB (%d unique ptrs / %d total uses)",
                    shape, sz, extra, extra * sz,
                    stats["count"], stats["total_layers"],
                )

    @classmethod
    def print_tape_comparison(cls):
        """Side-by-side comparison of tape tensor reuse across three modes.

        Modes:
          - *no-CG-free*: no-CG, _pack returns data_ptr (drops ref → allocator recycles)
          - *no-CG-keep*: no-CG, _pack returns tensor (keeps ref → fair baseline)
          - *CG*:         graph capture, _pack returns tensor (pool pins address)

        Categorizes each saved tensor as:
          - *activation*: ndim ≥ 3 (has batch/sequence dims)
          - *parameter*:  ndim ≤ 2 (weight/bias matrices)

        The key insight: if no-CG-keep shows similar low reuse to CG, the graph
        pool is NOT the root cause — all tapes are simply alive simultaneously.
        The gap between no-CG-free and no-CG-keep is the theoretical maximum
        recycling that freeing saved tensors would enable.
        """
        modes = [
            ("no-CG-free", cls._tape_ptr_history),
            ("no-CG-keep", cls._tape_nocg_keep_history),
            ("CG",         cls._tape_cg_history),
        ]
        if not any(h for _, h in modes):
            logger.info("TAPE-COMPARE: no data collected.")
            return

        def _build_shape_stats(history):
            """Return {shape: (unique_ptrs, total_uses, size_bytes)}."""
            ptr_to_count: Dict[int, int] = {}
            ptr_info: Dict[int, Tuple[Tuple[int, ...], int]] = {}
            for entries in history:
                for ptr, shape, size in entries:
                    ptr_to_count[ptr] = ptr_to_count.get(ptr, 0) + 1
                    if ptr not in ptr_info:
                        ptr_info[ptr] = (shape, size)
            shape_stats: Dict[Tuple[int, ...], List[int]] = {}
            for ptr, count in ptr_to_count.items():
                shape, size = ptr_info[ptr]
                if shape not in shape_stats:
                    shape_stats[shape] = [0, 0, size]
                shape_stats[shape][0] += 1
                shape_stats[shape][1] += count
            return shape_stats

        all_stats = {label: _build_shape_stats(hist) for label, hist in modes if hist}
        all_shapes = set()
        for s in all_stats.values():
            all_shapes |= set(s)
        all_shapes = sorted(all_shapes, key=lambda s: -max(
            all_stats[l].get(s, [0, 0, 0])[2] for l in all_stats))

        def _category(shape):
            return "activation" if len(shape) >= 3 else "parameter"

        # Per-category aggregates: {category: {mode: {unique, total, saved_mb}}}
        cat_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for cat in ("activation", "parameter"):
            cat_summary[cat] = {label: {"unique": 0, "total": 0, "saved_mb": 0.0}
                                for label, _ in modes if _}

        logger.info("TAPE-COMPARE: per-shape reuse (free=drop ref, keep=hold ref, CG=graph pool)")
        for shape in all_shapes:
            cat = _category(shape)
            sz_bytes = max(all_stats[l].get(shape, [0, 0, 0])[2] for l in all_stats)
            row = []
            for label, hist in modes:
                if not hist:
                    continue
                u, t, _ = all_stats[label].get(shape, (0, 0, sz_bytes))
                saved = (t - u) * sz_bytes / 1e6
                row.append((label, u, t, saved))
                cat_summary[cat][label]["unique"] += u
                cat_summary[cat][label]["total"] += t
                cat_summary[cat][label]["saved_mb"] += saved
            # Show shapes where keep vs CG differ (the real gap)
            keep_saved = next((s for l, _, _, s in row if l == "no-CG-keep"), 0)
            cg_saved = next((s for l, _, _, s in row if l == "CG"), 0)
            if abs(keep_saved - cg_saved) > 1.0 or cat == "parameter":
                parts = "  ".join(f"{l}: {u}/{t} ({s:.0f}MB)" for l, u, t, s in row)
                logger.info("  %-40s  %8.1fMB  %s  %s", str(shape), sz_bytes / 1e6, parts, cat)

        logger.info("TAPE-COMPARE: per-category summary")
        for cat in ("activation", "parameter"):
            parts = []
            for label, hist in modes:
                if not hist or label not in cat_summary[cat]:
                    continue
                s = cat_summary[cat][label]
                reuse_pct = 100 * (1 - s["unique"] / s["total"]) if s["total"] else 0
                parts.append(f"{label}: {s['unique']} unique/{s['total']} uses ({reuse_pct:.1f}% reuse, {s['saved_mb']:.0f} MB saved)")
            logger.info("  %-12s  %s", cat, "  |  ".join(parts))

    @classmethod
    def print_address_layout(cls):
        """Show per-layer address ranges to prove the pool is monotonic in CG.

        For each layer, prints the min/max data_ptr of its saved tensors and
        whether that range overlaps with any previous layer's range.

        Expected results:
          * no-CG: ranges overlap heavily (caching allocator recycles addresses)
          * CG: ranges are disjoint (graph pool pins each layer's addresses)

        Also categorizes each layer's saved tensors as:
          * *shared*: ptr appears in ≥2 layers (parameter-like)
          * *unique*: ptr appears in only 1 layer (activation, pool-pinned)
        """
        for label, history in [("no-CG-free", cls._tape_ptr_history),
                                ("no-CG-keep", cls._tape_nocg_keep_history),
                                ("CG", cls._tape_cg_history)]:
            if not history:
                continue

            # Build global ptr→layer-count map
            ptr_to_layers: Dict[int, Set[int]] = {}
            for idx, entries in enumerate(history):
                for ptr, _, _ in entries:
                    ptr_to_layers.setdefault(ptr, set()).add(idx)

            logger.info("ADDRESS-LAYOUT (%s): %d layers", label, len(history))
            layer_ranges: List[Tuple[int, int]] = []
            for idx, entries in enumerate(history):
                ptrs = [p for p, _, _ in entries]
                if not ptrs:
                    continue
                lo, hi = min(ptrs), max(ptrs)
                span_mb = (hi - lo + 1) / 1e6
                # Check overlap with previous layers
                overlap_layers: List[str] = []
                for prev_idx, (prev_lo, prev_hi) in enumerate(layer_ranges):
                    if lo <= prev_hi and hi >= prev_lo:
                        overlap_layers.append(f"L{prev_idx}")
                layer_ranges.append((lo, hi))

                # Count shared vs unique ptrs in this layer
                shared = sum(1 for p in ptrs if len(ptr_to_layers[p]) > 1)
                unique = len(ptrs) - shared
                unique_mb = sum(s for p, _, s in entries if len(ptr_to_layers[p]) == 1) / 1e6

                logger.info(
                    "  L%-3d  ptrs=%-4d (shared=%-4d unique=%-4d)  "
                    "range=[%x, %x]  span=%.1f MB  unique_mem=%.1f MB  overlap=%s",
                    idx, len(ptrs), shared, unique,
                    lo, hi, span_mb, unique_mb,
                    ", ".join(overlap_layers) if overlap_layers else "(none)",
                )

            # Total address footprint
            all_ptrs = [p for entries in history for p, _, _ in entries]
            if all_ptrs:
                total_lo, total_hi = min(all_ptrs), max(all_ptrs)
                total_span = (total_hi - total_lo + 1) / 1e6
                # Per-layer average span
                per_layer_spans = []
                for entries in history:
                    ptrs = [p for p, _, _ in entries]
                    if ptrs:
                        per_layer_spans.append((max(ptrs) - min(ptrs) + 1) / 1e6)
                avg_per_layer = sum(per_layer_spans) / len(per_layer_spans) if per_layer_spans else 0
                logger.info(
                    "  TOTAL: %d unique ptrs, footprint=%.1f MB, "
                    "avg_per_layer_span=%.1f MB, footprint/avg=%.1fx",
                    len(set(all_ptrs)), total_span, avg_per_layer,
                    total_span / avg_per_layer if avg_per_layer > 0 else 0,
                )
            logger.info("")

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
        if not self._tape_keep_experiment:
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
            if not self._tape_keep_experiment:
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
            if not self._tape_keep_experiment:
                self._fwd_graph = torch.cuda.CUDAGraph()
                self._fwd_graph.register_generator_state(gen)
            else:
                stream = torch.cuda.current_stream()

            stream.wait_stream(torch.cuda.current_stream())

            # -- memory tracking: before capture --
            torch.cuda.reset_peak_memory_stats()
            _alloc_before = torch.cuda.memory_allocated()
            _reserved_before = torch.cuda.memory_reserved()

            if self._tape_keep_experiment:
                out = self._module.forward(*sample_args, **sample_kwargs)
                # out = self._call_module(static_inputs)
                del self._static_inputs
            else:
                with torch.cuda.stream(stream):
                    if self._tape_reuse_experiment:
                        _tape_entries: List[Tuple[int, Tuple[int, ...], int]] = []
                        def _pack(t):
                            sz = t.nelement() * t.element_size()
                            _tape_entries.append((t.data_ptr(), tuple(t.shape), sz))
                            return t.data_ptr()
                        def _unpack(stored):
                            raise RuntimeError("unreachable")
                        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
                            out = self._call_module(static_inputs)
                        self._tape_ptr_history.append(_tape_entries)
                        self.print_tape_reuse_report()
                    elif self._tape_keep_experiment:
                        _keep_entries: List[Tuple[int, Tuple[int, ...], int]] = []
                        _hold = len(self._tape_nocg_keep_history) < self._tape_experiment_max_layers
                        def _pack(t):
                            sz = t.nelement() * t.element_size()
                            _keep_entries.append((t.data_ptr(), tuple(t.shape), sz))
                            if _hold:
                                self._kept_tape_tensors_all.append(t)
                            return t
                        def _unpack(t):
                            return t
                        with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
                            out = self._call_module(static_inputs)
                        self._tape_nocg_keep_history.append(_keep_entries)
                        self.print_tape_keep_report()
                    else:
                        with torch.cuda.graph(self._fwd_graph, pool=self._pool.pool, stream=stream):
                            out = self._call_module(static_inputs)
                        # _cg_entries: List[Tuple[int, Tuple[int, ...], int]] = []
                        # def _pack(t):
                        #     _cg_entries.append((t.data_ptr(), tuple(t.shape), t.nelement() * t.element_size()))
                        #     return t
                        # def _unpack(t):
                        #     return t
                        # with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
                        #     with torch.cuda.graph(self._fwd_graph, pool=self._pool.pool, stream=stream):
                        #         out = self._call_module(static_inputs)
                        # self._tape_cg_history.append(_cg_entries)
                        # self.print_tape_cg_report()
                    self.print_address_layout()

            # -- memory tracking: after capture --
            _alloc_after = torch.cuda.memory_allocated()
            _reserved_after = torch.cuda.memory_reserved()
            _peak_alloc = torch.cuda.max_memory_allocated()
            _peak_reserved = torch.cuda.max_memory_reserved()

            if self._tape_reuse_experiment:
                for p in self._module.parameters():
                    p.grad = None
                gc.collect()
                torch.cuda.empty_cache()
                self._fwd_captured = True
            else:
                self._record_output_structure(out)
                self._static_outputs = tuple(self._flatten_output(out))
                self._fwd_captured = True

                saved_info = inspect_autograd_tape(self._static_outputs)
                _tape_count = len(saved_info)
                _tape_mb = sum(s["size_bytes"] for s in saved_info) / 1e6

                logger.info(
                    "%s fwd-capture mem: alloc %+.1f MB (%d→%d)  "
                    "reserved %+.1f MB (%d→%d)  "
                    "peak_alloc %d MB  peak_reserved %d MB  "
                    "tape: %d tens / %.1f MB",
                    self._log_prefix,
                    (_alloc_after - _alloc_before) / 1e6,
                    _alloc_before // 1_000_000, _alloc_after // 1_000_000,
                    (_reserved_after - _reserved_before) / 1e6,
                    _reserved_before // 1_000_000, _reserved_after // 1_000_000,
                    _peak_alloc // 1_000_000, _peak_reserved // 1_000_000,
                    _tape_count, _tape_mb,
                )
                if self._tape_keep_experiment:
                    self._static_outputs = None

        finally:
            _restore_hooks_recursive(saved_hooks)

        return out

    # ------------------------------------------------------------------
    # Backward capture
    # ------------------------------------------------------------------

    def capture_backward(
        self, grad_outputs: Tuple[torch.Tensor, ...],
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

            flat_static_outputs = self._static_outputs
            outputs_for_capture = tuple(o for o in flat_static_outputs if o.requires_grad)
            inputs_for_capture = tuple(t for t in self._static_inputs if t.requires_grad)
            grad_outputs_for_capture = tuple(
                sg for sg, o in zip(static_grad_outputs, flat_static_outputs)
                if o.requires_grad
            )

            self._bwd_graph = torch.cuda.CUDAGraph()
            self._bwd_graph.register_generator_state(gen)

            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.cuda.graph(self._bwd_graph, pool=self._pool.pool, stream=stream):
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
            for t in self._static_inputs:
                if t.requires_grad:
                    static_grad_inputs.append(next(grad_iter))
                else:
                    static_grad_inputs.append(None)

            self._static_grad_inputs = tuple(static_grad_inputs)
            self._bwd_captured = True

            # Release owning refs to I/O tensors.  The backward graph is
            # already captured and its tape has been released by
            # retain_graph=False above.  make_weak_ref replaces each
            # tensor with a non-owning alias that still points to the
            # graph-pool data_ptr — the graph writes to the same address
            # on replay, so replay clones see fresh data.
            self._static_outputs = make_weak_ref(self._static_outputs)
            self._static_grad_inputs = make_weak_ref(self._static_grad_inputs)

        finally:
            _restore_hooks_recursive(saved_hooks)
            self._reshard_main_grad_buffer()

        # Run first real backward with live grads
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
        return self._module.forward(**kwargs)

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

    # ------------------------------------------------------------------
    # Tape memory sharing across layers (experiment — no CG)
    # ------------------------------------------------------------------

    @staticmethod
    def profile_tape_sharing(modules, sample_inputs, num_layers=None):
        """Run the same forward over multiple layers WITHOUT cuda graph,
        checking whether layer-0 tape addresses are reused by later layers.

        Returns a dict with per-layer tape tensors and reuse statistics.
        """
        import torch.distributed as dist

        num_layers = num_layers or len(modules)
        all_data_ptrs: List[Set[int]] = []
        all_tape_counts: List[int] = []
        all_tape_mb: List[float] = []

        for idx in range(num_layers):
            module = modules[idx]
            saved = _pop_hooks_recursive(module)
            try:
                out = module.forward(**dict(zip(
                    _get_forward_param_names(module.__class__), sample_inputs
                )))
                tape = inspect_autograd_tape((out,) if isinstance(out, torch.Tensor) else out)
                ptrs = {s["data_ptr"] for s in tape}
                all_data_ptrs.append(ptrs)
                all_tape_counts.append(len(tape))
                all_tape_mb.append(sum(s["size_bytes"] for s in tape) / 1e6)

                # Free the tape so next layer can reuse addresses
                loss = out.float().pow(2).mean()
                loss.backward()
                del out, loss, tape
                for p in module.parameters():
                    p.grad = None
            finally:
                _restore_hooks_recursive(saved)

            gc.collect()
            torch.cuda.empty_cache()

        # Compute per-address reuse: how many layers used each data_ptr
        ptr_to_layers: Dict[int, List[int]] = {}
        for idx, ptrs in enumerate(all_data_ptrs):
            for p in ptrs:
                ptr_to_layers.setdefault(p, []).append(idx)

        reused = sum(1 for layers in ptr_to_layers.values() if len(layers) > 1)
        total = len(ptr_to_layers)
        max_reuse = max((len(v) for v in ptr_to_layers.values()), default=0)

        if dist.get_rank() == 0:
            logger.info(
                "TAPE-REUSE: %d layers, %d total unique data_ptrs, "
                "%d reused across ≥2 layers (%.1f%%), max same-addr layers=%d. "
                "Tape per layer: %d tens / %.1f MB",
                num_layers, total, reused,
                100 * reused / total if total else 0,
                max_reuse,
                sum(all_tape_counts) // max(num_layers, 1),
                sum(all_tape_mb) / max(num_layers, 1),
            )
            # Show first reused ptr
            for p, layers in sorted(ptr_to_layers.items(), key=lambda x: -len(x[1]))[:3]:
                if len(layers) > 1:
                    logger.info("  ptr=%d reused by layers %s", p, layers[:8])

        return {
            "all_data_ptrs": all_data_ptrs,
            "ptr_to_layers": ptr_to_layers,
            "reused": reused,
            "total_unique": total,
            "tape_counts": all_tape_counts,
            "tape_mb": all_tape_mb,
        }

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
        self._fwd_captured = False
        self._bwd_captured = False
