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

Per-module split forward/backward CUDA graph capture with a SHARED memory
pool across modules. Designed to be a drop-in replacement for the original
`make_graphed_callables`-based runner that interleaved fwd+bwd capture.

Key design — split fwd/bwd capture for shared-pool safety
---------------------------------------------------------
The original `torch.cuda.make_graphed_callables` captures forward AND
backward of one callable BACK-TO-BACK. When you do this once per module
with a shared pool, the capture order looks like::

    capture: fwd1, bwd1, fwd2, bwd2, fwd3, bwd3
    runtime: fwd1, fwd2, fwd3, bwd3, bwd2, bwd1     ← MISMATCH → corruption

This runner instead captures forward and backward as TWO SEPARATE
``torch.cuda.CUDAGraph`` objects. Capture happens lazily during the
first eager forward/backward, driven by FSDP hooks (which fire in
execution order naturally). With one shared pool, capture order and
runtime order match::

    capture: fwd1, fwd2, fwd3, bwd3, bwd2, bwd1     ← driven by hook order
    runtime: fwd1, fwd2, fwd3, bwd3, bwd2, bwd1     ← MATCH ✓

Memory savings vs. private-pool-per-layer
------------------------------------------
* **Workspace sharing** — cuDNN / cuBLAS scratch buffers live in the
  pool and are reused across layers (instead of being duplicated N
  times in N private pools).
* **Pool packing** — the allocator packs allocations within one pool
  much more tightly than across N independent pools.

API
---
``FSDPCudaGraphRunner(fsdp_module, graph_pool=...)``
    * ``capture_forward(*args, **kwargs)`` — eager forward + capture
      forward graph. Must be called FIRST (drives capture order).
    * ``install()`` — patch ``module.forward`` to use a custom
      ``autograd.Function`` that replays the forward graph and, on
      backward, lazily captures + replays a backward graph.
    * ``uninstall()`` — restore the original ``forward``.

Backward capture is fully lazy: the first time autograd reaches the
custom ``Function.backward``, we run an eager backward to capture the
backward graph, then replay it. Subsequent microbatches replay both.
"""

import gc
import inspect
import logging
import os
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Debug toggles (env-var gated, no-op when disabled)
# ------------------------------------------------------------------
# MFSDP_CG_MEM_DEBUG selects a memory-debug mode. CG and no-CG are
# measured in SEPARATE runs to avoid mutual interference between the
# CUDA graph pool and the caching allocator:
#
#   MFSDP_CG_MEM_DEBUG=cg    — capture CG normally; for the first
#                              MFSDP_CG_MEM_SNAP_LAYERS layers, dump
#                              cg_layer{N}_rank{R}.pickle right after
#                              the forward graph is captured.
#   MFSDP_CG_MEM_DEBUG=nocg  — do NOT capture any CG. capture_forward
#                              just stores debug state; install()
#                              patches module.forward so the REAL
#                              forward (the one that runs after the
#                              pre_hook returns) is wrapped with
#                              memory recording + snapshot dump.
#                              Files: nocg_layer{N}_rank{R}.pickle.
#   MFSDP_CG_MEM_DEBUG=1     — alias for "cg".
#
# Snapshots are loadable at https://pytorch.org/memory_viz — drag in
# the cg and nocg pickles from two separate runs to compare.
_CG_MEM_MODE_RAW: str = os.environ.get("MFSDP_CG_MEM_DEBUG", "").lower()
if _CG_MEM_MODE_RAW in ("cg", "1", "true", "yes"):
    _CG_MEM_MODE: str = "cg"
elif _CG_MEM_MODE_RAW in ("nocg", "no-cg", "eager"):
    _CG_MEM_MODE = "nocg"
else:
    _CG_MEM_MODE = ""
# Optional override for the snapshot dump directory. Defaults to
# ./cg_mem_snapshots.
_CG_MEM_SNAPSHOT_DIR: Optional[str] = os.environ.get("MFSDP_CG_MEM_SNAPSHOT_DIR")
# Number of leading captured layers to dump per-layer torch.cuda memory
# snapshots for. Useful for checking whether the no-CG caching allocator
# reuses memory across layers while the CG graph pool does not. Set to a
# larger value to trace more layers (filenames are zero-indexed:
# cg_layer0_rankN.pickle, cg_layer1_rankN.pickle, ...). Default 2.
_CG_MEM_SNAP_LAYERS: int = int(os.environ.get("MFSDP_CG_MEM_SNAP_LAYERS", "2"))

# MFSDP_CG_COMPILE_FWD controls whether capture_forward compiles
# module.forward with torch.compile before warmup + capture. Without
# this, the captured graph runs the un-fused Python-level forward body
# (each matmul / activation allocates its own workspace and saves its
# own activation for backward), which uses significantly more memory
# than the no-CG path where the user's blk.compile() drives inductor
# fusion. With this enabled, the captured graph contains inductor
# triton kernels — same memory profile as the no-CG path.
#
#   MFSDP_CG_COMPILE_FWD=1   — compile forward body, capture fused kernels
#   MFSDP_CG_COMPILE_FWD=0   — (default) legacy: capture eager forward body
_CG_COMPILE_FWD: bool = os.environ.get("MFSDP_CG_COMPILE_FWD", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


# ------------------------------------------------------------------
# Hook helpers
# ------------------------------------------------------------------

# All known hook attributes across PyTorch versions (including 2.x additions).
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
    """Return the ordered parameter names of ``forward`` (excluding 'self')."""
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


def _pop_hooks(module: torch.nn.Module) -> Dict[str, Any]:
    """Remove all hooks from *module* (non-recursive) and return a snapshot."""
    saved: Dict[str, Any] = {}
    for attr in _HOOK_ATTRS:
        if hasattr(module, attr):
            saved[attr] = getattr(module, attr)
            setattr(module, attr, OrderedDict())
    return saved


def _pop_hooks_recursive(
    module: torch.nn.Module,
) -> List[Tuple[torch.nn.Module, Dict[str, Any]]]:
    """Remove all hooks from *module* and all its submodules recursively."""
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []
    for submodule in module.modules():
        saved.append((submodule, _pop_hooks(submodule)))
    return saved


def _restore_hooks(module: torch.nn.Module, saved: Dict[str, Any]) -> None:
    """Put the hooks back exactly as they were."""
    for name, value in saved.items():
        if value is not None:
            setattr(module, name, value)


def _restore_hooks_recursive(
    module: torch.nn.Module,
    saved: List[Tuple[torch.nn.Module, Dict[str, Any]]],
) -> None:
    """Restore hooks for all submodules saved by ``_pop_hooks_recursive``."""
    for submodule, sub_saved in saved:
        _restore_hooks(submodule, sub_saved)


# ------------------------------------------------------------------
# Custom autograd Function — replays fwd graph, lazily captures bwd graph
# ------------------------------------------------------------------


class _CudaGraphFunction(torch.autograd.Function):
    """Autograd Function that wires a fwd/bwd CUDA graph pair into autograd.

    Forward path
    ------------
    1. Copy live inputs into static input buffers.
    2. Replay the forward graph — writes into static output buffers.
    3. Return clones of the static outputs (so the autograd tape sees
       fresh leaf-like tensors and downstream ops never alias the
       graph's static output memory).

    Backward path
    -------------
    On first backward:
      * Capture the backward graph by running an eager backward with
        gradient inputs cloned into static grad-output buffers, while
        gradients are accumulated into static grad-input buffers.
    On subsequent backwards:
      * Copy upstream grads into static grad-output buffers, replay
        the backward graph, return static grad-inputs.
    """

    @staticmethod
    def forward(ctx, runner, *flat_inputs):
        # 1. Stage inputs into static buffers
        for static, live in zip(runner.static_inputs, flat_inputs):
            if static.data_ptr() != live.data_ptr():
                static.copy_(live)

        # 2. Replay the forward graph
        runner.fwd_graph.replay()

        # 3. Stash for backward
        ctx.runner = runner
        ctx.save_for_backward(*flat_inputs)

        # Return clones so downstream ops do not alias the static
        # output buffers (which the next forward replay overwrites).
        return tuple(o.clone() for o in runner.static_outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        runner = ctx.runner

        # ---- First backward: capture backward graph ----
        if runner.bwd_graph is None:
            return runner._capture_backward_and_run(ctx.saved_tensors, grad_outputs)

        # ---- Subsequent backwards: replay ----
        for static, live in zip(runner.static_grad_outputs, grad_outputs):
            if live is None:
                continue
            if static.data_ptr() != live.data_ptr():
                static.copy_(live)

        runner.bwd_graph.replay()

        # Return (None for runner) + clones of static grad-inputs
        return (None,) + tuple(
            None if g is None else g.clone()
            for g in runner.static_grad_inputs
        )


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------


class FSDPCudaGraphRunner:
    """Captures forward and backward CUDA graphs SEPARATELY for one
    FSDP module, sharing a pool with all other modules' graphs.

    Public API matches the original ``FSDPCudaGraphRunner``:

        runner = FSDPCudaGraphRunner(my_fsdp_module, graph_pool=pool)
        runner.capture_forward(*sample_args, **sample_kwargs)
        runner.install()
        # ... training loop runs eagerly through patched forward;
        #     first backward captures bwd graph, subsequent ones replay
        runner.uninstall()

    The ``graph_pool`` argument is REQUIRED (well, strongly recommended)
    when using multiple modules — pass the same handle to every module
    so all forward + backward graphs share one pool.

    Parameters
    ----------
    fsdp_module:
        The FSDP module to capture.
    graph_pool:
        Shared CUDA graph memory pool handle (from
        ``torch.cuda.graph_pool_handle()``). All FSDP modules sharing
        a pool MUST be captured in true execution order; the FSDP
        forward hook fires in execution order, so this naturally holds.
    gc_freeze:
        If True (default), call ``gc.collect()`` and ``gc.freeze()``
        before capture to prevent Python GC from stalling replay.
    capture_stream:
        Optional ``torch.cuda.Stream`` to use as the capture stream.
        When sharing a pool, all captures should run on the same
        stream — typically ``ctx.cuda_graph_stream``.
    num_warmup_iters:
        Eager warmup iterations before capture (default 3). Settles
        cuDNN benchmarking and TE FP8 scales.
    """

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        graph_pool: Optional[Any] = None,
        gc_freeze: bool = True,
        capture_stream: Optional[torch.cuda.Stream] = None,
        num_warmup_iters: int = 3,
    ):
        warnings.warn(
            "FSDPCudaGraphRunner is an experimental feature. The API and "
            "behaviour may change in future releases without notice.",
            FutureWarning,
            stacklevel=2,
        )

        self._module: torch.nn.Module = fsdp_module
        self._graph_pool: Optional[Any] = graph_pool
        self._gc_freeze: bool = gc_freeze
        self._capture_stream: Optional[torch.cuda.Stream] = capture_stream
        self._num_warmup_iters: int = num_warmup_iters

        # Module identifier for debug logging
        self._module_name = getattr(fsdp_module, "_fsdp_module_name", fsdp_module.__class__.__name__)
        self._log_prefix = f"[{self._module_name}]"

        # Forward graph state
        self.fwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_inputs: Tuple[torch.Tensor, ...] = ()
        self.static_outputs: Tuple[torch.Tensor, ...] = ()

        # Backward graph state (captured lazily on first backward)
        self.bwd_graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_grad_outputs: Tuple[torch.Tensor, ...] = ()
        self.static_grad_inputs: Tuple[Optional[torch.Tensor], ...] = ()

        # Frozen capture metadata
        self._tensor_param_names: List[str] = []
        self._param_names: List[str] = []
        self._frozen_kwargs: Dict[str, Any] = {}
        self._output_is_tuple: bool = True
        self._none_mask: Optional[List[bool]] = None

        # Install state
        self._orig_fwd: Optional[Any] = None
        self._captured: bool = False
        self._installed: bool = False

        # Inductor-fusion state for capture_forward. When
        # _CG_COMPILE_FWD is enabled we temporarily replace
        # self._module.forward with torch.compile(<original forward>)
        # so warmup populates dynamo's cache and capture runs the
        # cached triton kernels (which are CUDA-graph capturable).
        # After capture we restore the original forward so install()
        # sees the user-written body.
        self._captured_fwd_was_compiled: bool = False
        self._orig_fwd_body: Optional[Any] = None

        # Debug state (nocg mode — patches module.forward to record the
        # REAL forward call instead of running a separate debug forward)
        self._debug_layer_index: int = -1
        self._debug_do_snapshot: bool = False
        self._debug_snap_dir: str = ""
        self._debug_rank: int = 0
        self._debug_recorded: bool = False

    # ------------------------------------------------------------------
    # 1. Forward capture
    # ------------------------------------------------------------------

    def capture_forward(self, *sample_args, **sample_kwargs) -> None:
        """Eagerly warm up + capture the forward graph.

        Runs ``num_warmup_iters`` eager forward+backward passes (no
        capture) to settle cuDNN / FP8 scales, then captures one
        forward pass into ``self.fwd_graph`` using the shared pool.
        """
        assert getattr(self._module, "cuda_graph_compatible", True), (
            "CUDA graph capture requires TracePoolAllocator in optimized phase"
        )

        # ---- 1. Introspect signature, separate tensor / non-tensor inputs ----
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

        # ---- 2. Build static input buffers ----
        # Static buffers are clones of the live samples, allocated INSIDE
        # the shared pool's address space because we are about to enter
        # graph capture. They become the addresses recorded in the graph.
        flat_live = tuple(bound[n] for n in tensor_names)
        # We hold a clone for static inputs that requires_grad mirrors live.
        static_inputs = tuple(
            t.clone().detach().requires_grad_(t.requires_grad) for t in flat_live
        )

        # Zero grads, unshard main grad buffer (matches original runner).
        for param in self._module.parameters():
            param.grad = None
        self._unshard_main_grad_buffer()

        ctx = getattr(self._module, "_fsdp_root_context", None)

        # Debug: detect the first N captured layers so we can dump a
        # torch memory snapshot for each. The capture pre-hook fires in
        # forward execution order, so the counter matches the layer
        # index (0-based). Snapshots are controlled by
        # MFSDP_CG_MEM_SNAP_LAYERS (default 2: traces layer0 + layer1
        # so we can see whether the no-CG caching allocator reuses
        # memory between layers while the CG graph pool does not).
        layer_index = -1
        do_snapshot = False
        if _CG_MEM_MODE and ctx is not None:
            seq = getattr(ctx, "_cg_capture_seq", 0)
            layer_index = seq
            do_snapshot = seq < _CG_MEM_SNAP_LAYERS
            ctx._cg_capture_seq = seq + 1

        # ============================================================
        # nocg mode: store debug state and return. install() will patch
        # module.forward so the REAL forward (after pre_hook returns)
        # is recorded + snapshotted. No hooks-pop, no warmup, no extra
        # forward — the snapshot captures the actual training forward.
        # ============================================================
        if _CG_MEM_MODE == "nocg":
            _rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            _snap_dir = _CG_MEM_SNAPSHOT_DIR or "cg_mem_snapshots"
            self._debug_layer_index = layer_index
            self._debug_do_snapshot = do_snapshot
            self._debug_snap_dir = _snap_dir
            self._debug_rank = _rank
            self._debug_recorded = False
            self._captured = True
            return

        # ---- 3. Pop hooks so capture sees only forward() body ----
        saved_hooks = _pop_hooks_recursive(self._module)
        prev_active = False
        if ctx is not None:
            prev_active = getattr(ctx, "cuda_graph_active", False)
            ctx.cuda_graph_active = True

        # GC freeze
        if self._gc_freeze:
            gc.collect()
            gc.freeze()

        try:
            # ---- 3b. Optionally compile forward body so capture contains
            #          inductor-fused triton kernels (not eager ops) ----
            # See _CG_COMPILE_FWD docstring. We replace self._module.forward
            # with torch.compile(orig) so the warmup below populates
            # dynamo's cache; the subsequent capture (inside
            # torch.cuda.graph) runs the cached compiled code, whose
            # triton kernels are capturable. Without this, capture runs
            # the un-fused python body, which uses significantly more
            # memory than the no-CG path.
            if _CG_COMPILE_FWD:
                _orig_fwd_body = self._module.forward
                # Skip if the user already compiled the forward directly.
                if not hasattr(_orig_fwd_body, "get_compiler_config"):
                    try:
                        self._module.forward = torch.compile(_orig_fwd_body)
                        self._captured_fwd_was_compiled = True
                        self._orig_fwd_body = _orig_fwd_body
                        logger.info(
                            "%s [cg-compile-fwd] compiled forward body for "
                            "inductor fusion during capture",
                            self._log_prefix,
                        )
                    except Exception as e:
                        logger.warning(
                            "%s [cg-compile-fwd] torch.compile failed (%s); "
                            "capturing eager forward (legacy behavior)",
                            self._log_prefix,
                            e,
                        )

            # ---- 4. Pick / create capture stream ----
            capture_stream = self._capture_stream
            if capture_stream is None:
                capture_stream = torch.cuda.Stream()

            # ---- 5. Warmup on a side stream (matches PyTorch's own
            #          make_graphed_callables) ----
            torch.cuda.synchronize()
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(self._num_warmup_iters):
                    out = self._call_module(static_inputs, tensor_names, frozen_kwargs)
                    flat_out = self._flatten_output_for_autograd(out)
                    if any(o.requires_grad for o in flat_out):
                        grads = tuple(
                            torch.empty_like(o) for o in flat_out if o.requires_grad
                        )
                        torch.autograd.grad(
                            outputs=tuple(o for o in flat_out if o.requires_grad),
                            inputs=tuple(
                                t for t in static_inputs + tuple(self._module.parameters())
                                if t.requires_grad
                            ),
                            grad_outputs=grads,
                            only_inputs=True,
                            allow_unused=True,
                            retain_graph=False,
                        )
                    # Drop references so warmup activations free.
                    del out, flat_out
            torch.cuda.current_stream().wait_stream(warmup_stream)
            torch.cuda.synchronize()

            # Reset grads after warmup
            for param in self._module.parameters():
                param.grad = None

            # ---- Common debug helpers ----
            _rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            _snap_dir = _CG_MEM_SNAPSHOT_DIR or "cg_mem_snapshots"

            # ============================================================
            # cg mode: capture the forward graph
            # ============================================================
            # Start stack-aware recording before capture so the CG
            # snapshot (first N layers) carries allocation call stacks.
            _cg_record = _CG_MEM_MODE == "cg" and do_snapshot
            if _cg_record:
                os.makedirs(_snap_dir, exist_ok=True)
                torch.cuda.memory._record_memory_history(
                    max_entries=200000, stacks="all"
                )
            # -- memory tracking: before capture --
            torch.cuda.reset_peak_memory_stats()
            _alloc_before = torch.cuda.memory_allocated()
            _reserved_before = torch.cuda.memory_reserved()

            # ---- 6. Capture forward graph on the shared pool ----
            self.fwd_graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(capture_stream):
                with torch.cuda.graph(
                    self.fwd_graph,
                    pool=self._graph_pool,
                    stream=capture_stream,
                ):
                    out = self._call_module(static_inputs, tensor_names, frozen_kwargs)

            # -- memory tracking: after capture --
            _alloc_after = torch.cuda.memory_allocated()
            _reserved_after = torch.cuda.memory_reserved()
            _peak_alloc = torch.cuda.max_memory_allocated()
            _peak_reserved = torch.cuda.max_memory_reserved()

            logger.info(
                "%s fwd-capture mem: alloc %+.1f MB (%d→%d)  "
                "reserved %+.1f MB (%d→%d)  "
                "peak_alloc %d MB  peak_reserved %d MB  ",
                self._log_prefix,
                (_alloc_after - _alloc_before) / 1e6,
                _alloc_before // 1_000_000, _alloc_after // 1_000_000,
                (_reserved_after - _reserved_before) / 1e6,
                _reserved_before // 1_000_000, _reserved_after // 1_000_000,
                _peak_alloc // 1_000_000, _peak_reserved // 1_000_000,
            )

            # -- debug: aligned per-layer CG peak log --
            if _CG_MEM_MODE == "cg":
                logger.info(
                    "%s [mem-debug] cg peak memory (MB): "
                    "peak_alloc %.1f (delta %+.1f)  "
                    "peak_reserved %.1f (delta %+.1f)  post %d MB",
                    self._log_prefix,
                    _peak_alloc / 1e6,
                    (_peak_alloc - _alloc_before) / 1e6,
                    _peak_reserved / 1e6,
                    (_reserved_after - _reserved_before) / 1e6,
                    _alloc_after // 1_000_000,
                )

            # -- debug: dump CG snapshot for the first N layers --
            if _cg_record:
                try:
                    _cg_path = os.path.join(
                        _snap_dir, f"cg_layer{layer_index}_rank{_rank}.pickle"
                    )
                    torch.cuda.memory._dump_snapshot(_cg_path)
                    logger.info(
                        "%s [mem-debug] dumped cg snapshot (layer %d): %s",
                        self._log_prefix,
                        layer_index,
                        _cg_path,
                    )
                except Exception as e:
                    logger.warning(
                        "%s [mem-debug] CG snapshot dump failed (layer %d): %s",
                        self._log_prefix,
                        layer_index,
                        e,
                    )
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)

            # Snapshot output structure (for None restoration on replay).
            self._record_output_structure(out)
            static_outputs_list = list(self._flatten_output_for_autograd(out))
            self.static_outputs = tuple(static_outputs_list)

        finally:
            if ctx is not None:
                ctx.cuda_graph_active = prev_active
            # Restore the original forward body if we replaced it with a
            # torch.compile call for capture. install() reads
            # self._module.forward immediately after this and expects the
            # user-written body (not a compiled wrapper).
            if self._captured_fwd_was_compiled:
                self._module.forward = self._orig_fwd_body
                self._orig_fwd_body = None
                self._captured_fwd_was_compiled = False
            _restore_hooks_recursive(self._module, saved_hooks)
            self._reshard_main_grad_buffer()
            if self._gc_freeze:
                try:
                    gc.unfreeze()
                except Exception:
                    pass

        # ---- 7. Save metadata ----
        self.static_inputs = static_inputs
        self._tensor_param_names = tensor_names
        self._param_names = param_names
        self._frozen_kwargs = frozen_kwargs
        self._captured = True

    # ------------------------------------------------------------------
    # 1b. Debug: recorded forward (nocg mode)
    # ------------------------------------------------------------------

    def _debug_recorded_forward(self, *args, **kwargs):
        """Debug wrapper around the real ``module.forward``.

        On the first call (step 0 of this layer), records
        stack-aware allocation history and dumps a
        ``nocg_layer{N}_rank{R}.pickle`` snapshot right after the
        forward returns — while the autograd tape is still intact.

        Subsequent calls are pass-through (no recording overhead).
        """
        do_snapshot = (
            self._debug_do_snapshot
            and not self._debug_recorded
        )
        recording = do_snapshot
        if recording:
            os.makedirs(self._debug_snap_dir, exist_ok=True)
            torch.cuda.memory._record_memory_history(
                max_entries=200000, stacks="all"
            )
        try:
            torch.cuda.reset_peak_memory_stats()
            _alloc_before = torch.cuda.memory_allocated()
            _reserved_before = torch.cuda.memory_reserved()

            out = self._orig_fwd(*args, **kwargs)

            _peak_alloc = torch.cuda.max_memory_allocated()
            _peak_reserved = torch.cuda.max_memory_reserved()
            _alloc_after = torch.cuda.memory_allocated()
            _reserved_after = torch.cuda.memory_reserved()

            if recording:
                try:
                    _path = os.path.join(
                        self._debug_snap_dir,
                        f"nocg_layer{self._debug_layer_index}_rank{self._debug_rank}.pickle",
                    )
                    torch.cuda.memory._dump_snapshot(_path)
                    logger.info(
                        "%s [mem-debug] dumped nocg snapshot (layer %d): %s",
                        self._log_prefix,
                        self._debug_layer_index,
                        _path,
                    )
                except Exception as e:
                    logger.warning(
                        "%s [mem-debug] nocg snapshot dump failed (layer %d): %s",
                        self._log_prefix,
                        self._debug_layer_index,
                        e,
                    )
        finally:
            if recording:
                torch.cuda.memory._record_memory_history(enabled=None)
                self._debug_recorded = True

        if do_snapshot:
            logger.info(
                "%s [mem-debug] nocg peak memory (MB) [layer %d]: "
                "peak_alloc %.1f (delta %+.1f)  "
                "peak_reserved %.1f (delta %+.1f)  post %d MB",
                self._log_prefix,
                self._debug_layer_index,
                _peak_alloc / 1e6,
                (_peak_alloc - _alloc_before) / 1e6,
                _peak_reserved / 1e6,
                (_reserved_after - _reserved_before) / 1e6,
                _alloc_after // 1_000_000,
            )

        return out

    # ------------------------------------------------------------------
    # 2. Backward graph capture (lazy, called from autograd Function)
    # ------------------------------------------------------------------

    def _capture_backward_and_run(
        self,
        saved_inputs: Tuple[torch.Tensor, ...],
        grad_outputs: Tuple[torch.Tensor, ...],
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """Capture the backward graph on first invocation.

        Strategy:
          1. Allocate static grad-output buffers (cloned from live).
          2. Re-run the forward eagerly (so we have a fresh autograd
             graph rooted at the same static_inputs) on the capture
             stream — this allocates inside the pool.
          3. Capture an eager backward into self.bwd_graph, populating
             static grad-input buffers.

        Returns the gradient inputs to feed back into autograd.
        """
        ctx = getattr(self._module, "_fsdp_root_context", None)
        prev_active = False
        if ctx is not None:
            prev_active = getattr(ctx, "cuda_graph_active", False)
            ctx.cuda_graph_active = True

        # Pop hooks so capture only sees backward()
        saved_hooks = _pop_hooks_recursive(self._module)

        # Unshard main grad buffer for capture
        self._unshard_main_grad_buffer()

        try:
            # 1. Static grad-output buffers — clone live grads
            static_grad_outputs = tuple(
                torch.zeros_like(o) if g is None else g.clone().detach()
                for o, g in zip(self.static_outputs, grad_outputs)
            )

            capture_stream = self._capture_stream or torch.cuda.current_stream()

            # 2. Re-run forward eagerly to build a backward graph rooted
            #    at static_inputs. This must run on the capture stream
            #    AND inside the same pool so addresses match.
            #
            #    We DO NOT wrap this in `torch.cuda.graph(...)` — we want
            #    the autograd graph to exist in Python-land. Then we wrap
            #    only the .backward() call inside graph capture.
            #
            #    To keep allocations inside the pool we use the
            #    `torch.cuda.graph` context twice: once with NO graph
            #    (effectively just stream switch) is not enough — we
            #    instead capture the backward directly while autograd's
            #    saved tensors live in regular memory and gradient
            #    allocations land in the pool.
            torch.cuda.synchronize()

            # Reset param.grad fields so AccumulateGrad nodes write
            # into freshly allocated tensors inside the pool.
            for param in self._module.parameters():
                param.grad = None

            # Re-run forward eagerly so we have an autograd tape rooted
            # at static_inputs producing static_outputs (or rather a
            # parallel tape of the same shapes).
            with torch.cuda.stream(capture_stream):
                # Make inputs require grad like at capture time
                replay_inputs = tuple(
                    t.detach().clone().requires_grad_(t.requires_grad)
                    for t in self.static_inputs
                )
                replay_out = self._call_module(
                    replay_inputs, self._tensor_param_names, self._frozen_kwargs
                )
                flat_replay_out = self._flatten_output_for_autograd(replay_out)

                # 3. Capture the backward
                self.bwd_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    self.bwd_graph,
                    pool=self._graph_pool,
                    stream=capture_stream,
                ):
                    grad_ins = torch.autograd.grad(
                        outputs=tuple(
                            o for o in flat_replay_out if o.requires_grad
                        ),
                        inputs=tuple(
                            t for t in replay_inputs if t.requires_grad
                        ),
                        grad_outputs=tuple(
                            sg for sg, o in zip(static_grad_outputs, flat_replay_out)
                            if o.requires_grad
                        ),
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                        allow_unused=True,
                    )

                # Build static_grad_inputs aligned with self.static_inputs
                # (with None where requires_grad=False).
                grad_iter = iter(grad_ins)
                static_grad_inputs: List[Optional[torch.Tensor]] = []
                for t in replay_inputs:
                    if t.requires_grad:
                        static_grad_inputs.append(next(grad_iter))
                    else:
                        static_grad_inputs.append(None)

                self.static_grad_outputs = static_grad_outputs
                self.static_grad_inputs = tuple(static_grad_inputs)

        finally:
            if ctx is not None:
                ctx.cuda_graph_active = prev_active
            _restore_hooks_recursive(self._module, saved_hooks)
            self._reshard_main_grad_buffer()

        # Now that capture is done, the FIRST backward also needs to
        # actually compute correct gradients for the live grad_outputs.
        # We do this by replaying the bwd graph with live grads.
        for static, live in zip(self.static_grad_outputs, grad_outputs):
            if live is None:
                continue
            if static.data_ptr() != live.data_ptr():
                static.copy_(live)
        self.bwd_graph.replay()

        return (None,) + tuple(
            None if g is None else g.clone()
            for g in self.static_grad_inputs
        )

    # ------------------------------------------------------------------
    # 3. Install / uninstall patched forward
    # ------------------------------------------------------------------

    def install(self) -> None:
        """Patch ``module.forward``.

        In normal (CG) mode, replaces forward with a custom autograd
        Function that replays the captured graph. In nocg debug mode,
        wraps forward with a memory-recording + snapshot wrapper.
        """
        if self._installed:
            return

        if _CG_MEM_MODE == "nocg":
            self._orig_fwd = self._module.forward
            runner = self

            def _patched_fwd(*args, **kwargs):
                return runner._debug_recorded_forward(*args, **kwargs)

            self._module.forward = _patched_fwd
            self._installed = True
            return

        if not self._captured:
            raise RuntimeError("Call capture_forward() first")

        self._orig_fwd = self._module.forward
        runner = self
        param_names = self._param_names
        tensor_names = self._tensor_param_names

        def _patched_fwd(*args, **kwargs):
            # Re-bind args/kwargs into the same flat tensor order used at
            # capture.
            bound: Dict[str, Any] = {}
            for i, val in enumerate(args):
                if i < len(param_names):
                    bound[param_names[i]] = val
            bound.update(kwargs)
            flat = tuple(bound[n] for n in tensor_names)

            outs = _CudaGraphFunction.apply(runner, *flat)
            return runner._unflatten_output_for_user(outs)

        self._module.forward = _patched_fwd
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._module.forward = self._orig_fwd
        self._orig_fwd = None
        self._installed = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def using_cuda_graph(self) -> bool:
        return self._installed

    def reset(self) -> None:
        self.uninstall()
        self.fwd_graph = None
        self.bwd_graph = None
        self.static_inputs = ()
        self.static_outputs = ()
        self.static_grad_outputs = ()
        self.static_grad_inputs = ()
        self._captured = False

    # ------------------------------------------------------------------
    # FSDP integration helpers
    # ------------------------------------------------------------------

    def _unshard_main_grad_buffer(self) -> None:
        for group in getattr(self._module, "_fsdp_param_groups", []):
            if hasattr(group, "main_grad_buffer") and group.main_grad_buffer is not None:
                group.main_grad_buffer.fetch_buffer()

    def _reshard_main_grad_buffer(self) -> None:
        for group in getattr(self._module, "_fsdp_param_groups", []):
            if hasattr(group, "release_grad_buffer"):
                group.release_grad_buffer()

    # ------------------------------------------------------------------
    # Module call / output (un)flattening helpers
    # ------------------------------------------------------------------

    def _call_module(
        self,
        flat_tensor_inputs: Tuple[torch.Tensor, ...],
        tensor_names: List[str],
        frozen_kwargs: Dict[str, Any],
    ) -> Any:
        kwargs = dict(zip(tensor_names, flat_tensor_inputs))
        kwargs.update(frozen_kwargs)
        return self._module.forward(**kwargs)

    def _record_output_structure(self, out: Any) -> None:
        """Snapshot whether output is a tuple, single tensor, and the
        positions of None entries (for restoration on replay)."""
        if isinstance(out, torch.Tensor):
            self._output_is_tuple = False
            self._none_mask = None
            return
        if isinstance(out, (tuple, list)):
            self._output_is_tuple = True
            self._none_mask = [t is None for t in out]
            return
        raise RuntimeError(
            f"Module returned unsupported output type: {type(out)}. "
            "CUDA graph capture supports a Tensor or tuple/list of Tensors/None."
        )

    def _flatten_output_for_autograd(self, out: Any) -> Tuple[torch.Tensor, ...]:
        """Return only the non-None tensors of *out* in declaration order."""
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(t for t in out if isinstance(t, torch.Tensor))

    def _unflatten_output_for_user(
        self, flat: Tuple[torch.Tensor, ...]
    ) -> Any:
        """Rebuild user-facing output from a flat tuple of tensors,
        re-inserting None at recorded positions and unwrapping
        single-tensor outputs."""
        if not self._output_is_tuple:
            return flat[0]

        if self._none_mask is None:
            return flat

        full: List[Any] = []
        it = iter(flat)
        for is_none in self._none_mask:
            full.append(None if is_none else next(it))
        return tuple(full)
