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

"""Batch CUDA graph capture helper for FSDP v2 — TECudaGraphHelper-compatible.

Captures all cuda-graph-enabled FSDP modules in a single
``torch.cuda.make_graphed_callables`` invocation with a **shared**
CUDA graph memory pool, mirroring the architecture of
``TECudaGraphHelper`` (see ``megatron/core/transformer/cuda_graphs.py``).

Key design:
* **Trace‑phase recording** — a lightweight forward pre‑hook fires
  before FSDP hooks during the trace (first) forward‑backward pass,
  recording every graphable module's tensor input shapes in the exact
  execution order observed by the FSDP v2 runtime.  This makes the
  helper compatible with *any* FSDP‑wrapped module, not just layers
  that expose ``get_layer_static_inputs()``.
* **Single ``make_graphed_callables``** — all modules are captured
  together with a shared ``torch.cuda.graph_pool_handle()``, so the
  CUDA driver packs the graphs into one pool instead of N private pools.
* **TE‑compatible API** — ``create_cudagraphs()``,
  ``capture_finished()``, ``cuda_graph_set_manual_hooks()``,
  ``delete_cuda_graphs()``.

Usage (in training loop, after ``cuda_graph_warmup_steps`` iterations)::

    helper = FSDPCudaGraphHelper(
        model=model,
        config=config,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
    )
    helper.start_trace()
    # … one forward‑backward step (trace phase) …
    helper.stop_trace()
    helper.create_cudagraphs()
    # … subsequent steps replay graphs …
    helper.delete_cuda_graphs()
"""

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.transformer.transformer_config import TransformerConfig

from .cuda_graph_runner import (
    _ForwardShim,
    _get_forward_param_names,
    _pop_hooks_recursive,
    _restore_hooks,
)

try:
    from transformer_engine.pytorch.graph import make_graphed_callables as te_make_graphed_callables
    HAVE_TE_GRAPHS = True
except ImportError:
    HAVE_TE_GRAPHS = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trace record
# ---------------------------------------------------------------------------


class _TraceRecord:
    """Snapshot of one forward call seen during the trace phase."""

    __slots__ = ("module", "tensor_names", "shapes", "frozen_kwargs")

    def __init__(
        self,
        module: torch.nn.Module,
        tensor_names: List[str],
        shapes: List[Tuple[torch.Size, torch.dtype, torch.device]],
        frozen_kwargs: Dict[str, Any],
    ) -> None:
        self.module = module
        self.tensor_names = list(tensor_names)
        self.shapes = list(shapes)  # (shape, dtype, device) per tensor arg
        self.frozen_kwargs = dict(frozen_kwargs)


# ---------------------------------------------------------------------------
# Microbatch tracker
# ---------------------------------------------------------------------------


def set_fsdp_current_microbatch(model: torch.nn.Module, microbatch_id: int) -> None:
    """Set ``current_microbatch`` on every FSDP module that has CUDA graphs.

    Called from the pipeline-parallel schedule before each forward step so
    that the patched forward selects the correct per-microbatch graph.
    """
    for module in model.modules():
        if hasattr(module, "_fsdp_cuda_graphs"):
            module.current_microbatch = microbatch_id


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


class FSDPCudaGraphHelper:
    """Batch CUDA graph capture for FSDP v2, modelled after ``TECudaGraphHelper``.

    Parameters:
        model:
            The FSDP‑wrapped model (a single ``torch.nn.Module``).
        config:
            ``TransformerConfig`` matching the training configuration.
        seq_length:
            Sequence length (used only when a module lacks a trace record
            and falls back to ``get_layer_static_inputs``).
        micro_batch_size:
            Micro‑batch size (same fallback use as *seq_length*).
        optimizers:
            (optional) List of optimizers; their gradients are zeroed after
            capture.
        pg_collection:
            (optional) ``ProcessGroupCollection``.  PP>1 disables
            single‑microbatch optimisation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: TransformerConfig,
        seq_length: int,
        micro_batch_size: int,
        optimizers: Optional[List[torch.optim.Optimizer]] = None,
        pg_collection: Optional[Any] = None,
    ) -> None:
        self.model: torch.nn.Module = model
        self.config: TransformerConfig = config
        self.seq_length: int = seq_length
        self.micro_batch_size: int = micro_batch_size
        self.optimizers: List[torch.optim.Optimizer] = optimizers or []

        self.pg_collection: Optional[Any] = pg_collection
        self.pp_size: int = 0
        if pg_collection is not None:
            self.pp_size = pg_collection.pp.size()

        # Trace state.
        self._phase: str = "init"  # init → trace → capture → replay
        self._trace_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._trace_records: List[_TraceRecord] = []

        # Discovered graphable modules (populated in start_trace).
        self.flattened_callables: List[torch.nn.Module] = []

        # Multi‑microbatch.
        self.num_microbatches: int = 1

        # Shared pool (created in create_cudagraphs).
        self._shared_pool: Optional[int] = None

        # Flags matching TECudaGraphHelper.
        self._capture_finished: bool = False
        self._graphs_created: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture_finished(self) -> bool:
        """``True`` after ``create_cudagraphs()`` has returned."""
        return self._capture_finished

    def graphs_created(self) -> bool:
        """``True`` if at least one CUDA graph was successfully created."""
        return self._graphs_created

    def cuda_graph_set_manual_hooks(self) -> None:
        """No‑op for FSDP v2.

        FSDP v2 manages its own hooks via
        ``_register_forward_pre_hook`` / ``_register_forward_hook``.
        """

    # ------------------------------------------------------------------
    # Trace phase — record execution order & input shapes
    # ------------------------------------------------------------------

    def start_trace(self) -> None:
        """Register a lightweight forward pre‑hook on every graphable
        FSDP module so that the first forward‑backward pass records
        the exact execution order and tensor input shapes.

        Must be called **before** the trace‑phase forward pass.
        """
        assert self._phase == "init", (
            f"start_trace called in phase '{self._phase}' — expected 'init'"
        )
        self._phase = "trace"

        # Discover graphable modules from the FSDP v2 forward order.
        ctx = self._get_root_context()
        forward_order = ctx.forward_order

        self.flattened_callables = [
            m
            for m in forward_order
            if (
                hasattr(m, "_fsdp_state")
                and m._fsdp_state.enable_cuda_graph
                and m._fsdp_state._is_leaf
            )
        ]

        if not self.flattened_callables:
            logger.info("FSDPCudaGraphHelper: no graphable FSDP modules found.")
            return

        # PP size determines microbatches.
        if self.pp_size <= 1:
            self.num_microbatches = 1
        else:
            self.num_microbatches = get_num_microbatches()

        logger.info(
            "FSDPCudaGraphHelper: tracing %d graphable FSDP modules "
            "(%d microbatches).",
            len(self.flattened_callables),
            self.num_microbatches,
        )

        # Register trace hooks — fire BEFORE FSDP hooks so we see raw inputs.
        for module in self.flattened_callables:
            handle = module.register_forward_pre_hook(
                self._trace_pre_hook, prepend=True, with_kwargs=True
            )
            self._trace_handles.append(handle)

    def _trace_pre_hook(self, hook_module, args, kwargs) -> None:
        """Record tensor shapes and names from the incoming forward args."""
        param_names = _get_forward_param_names(hook_module.__class__)
        tensor_names: List[str] = []
        shapes: List[Tuple[torch.Size, torch.dtype, torch.device]] = []
        frozen_kwargs: Dict[str, Any] = {}

        bound: Dict[str, Any] = {}
        for i, val in enumerate(args):
            if i < len(param_names):
                bound[param_names[i]] = val
        bound.update(kwargs)

        for pn in param_names:
            if pn not in bound:
                continue
            val = bound[pn]
            if isinstance(val, torch.Tensor):
                tensor_names.append(pn)
                shapes.append((val.shape, val.dtype, val.device))
            else:
                frozen_kwargs[pn] = val

        self._trace_records.append(
            _TraceRecord(
                module=hook_module,
                tensor_names=tensor_names,
                shapes=shapes,
                frozen_kwargs=frozen_kwargs,
            )
        )

    def stop_trace(self) -> None:
        """Unregister all trace hooks.  Call after the first forward‑backward
        pass completes and the allocator has entered the *optimized* phase.
        """
        assert self._phase == "trace", (
            f"stop_trace called in phase '{self._phase}' — expected 'trace'"
        )
        for handle in self._trace_handles:
            handle.remove()
        self._trace_handles.clear()

    # ------------------------------------------------------------------
    # Build shims + sample args from trace records
    # ------------------------------------------------------------------

    def _build_from_trace(
        self,
    ) -> Tuple[List[_ForwardShim], List[Tuple[torch.Tensor, ...]]]:
        """Convert trace records into shims and sample‑arg tuples.

        Each (module, microbatch) pair gets its own shim and sample args.
        Modules with missing trace records fall back to
        ``get_layer_static_inputs()``.

        Returns:
            (shims, sample_args):
                Two parallel lists, each of length
                ``len(flattened_callables) * num_microbatches``.
        """
        shims: List[_ForwardShim] = []
        sample_args: List[Tuple[torch.Tensor, ...]] = []

        # Index trace records by module id for O(1) lookup.
        record_by_id: Dict[int, _TraceRecord] = {}
        for rec in self._trace_records:
            record_by_id[id(rec.module)] = rec

        for module in self.flattened_callables:
            rec = record_by_id.get(id(module))

            if rec is not None:
                tensor_names = list(rec.tensor_names)
                frozen_kwargs = dict(rec.frozen_kwargs)
                shapes = list(rec.shapes)
            else:
                # Fallback: use static inputs.  This path is taken for
                # modules whose forward was never called during trace
                # (e.g. layers on a different pipeline stage).
                tensor_names, frozen_kwargs, shapes = self._fallback_static_inputs(
                    module
                )

            for _ in range(self.num_microbatches):
                shim = _ForwardShim(module, list(tensor_names), dict(frozen_kwargs))
                shims.append(shim)

                flat_sample = tuple(
                    torch.zeros(shape, dtype=dtype, device=device).requires_grad_(True)
                    for shape, dtype, device in shapes
                )
                sample_args.append(flat_sample)

        return shims, sample_args

    def _fallback_static_inputs(
        self, module: torch.nn.Module
    ) -> Tuple[List[str], Dict[str, Any], List[Tuple[torch.Size, torch.dtype, torch.device]]]:
        """Generate sample metadata via ``get_layer_static_inputs()``."""
        if not hasattr(module, "get_layer_static_inputs"):
            raise RuntimeError(
                f"Module {module.__class__.__name__} has no trace record and "
                f"no get_layer_static_inputs() — cannot generate sample inputs."
            )

        param_names = _get_forward_param_names(module.__class__)
        static_inputs = module.get_layer_static_inputs(
            self.seq_length, self.micro_batch_size
        )

        tensor_names: List[str] = []
        frozen_kwargs: Dict[str, Any] = {}
        shapes: List[Tuple[torch.Size, torch.dtype, torch.device]] = []

        for pn in param_names:
            if pn not in static_inputs:
                continue
            val = static_inputs[pn]
            if isinstance(val, torch.Tensor):
                tensor_names.append(pn)
                shapes.append((val.shape, val.dtype, val.device))
            else:
                frozen_kwargs[pn] = val

        return tensor_names, frozen_kwargs, shapes

    # ------------------------------------------------------------------
    # Capture orchestration
    # ------------------------------------------------------------------

    def _get_root_context(self):
        """Return the shared ``_FSDPRootContext`` from the model."""
        for module in self.model.modules():
            if hasattr(module, "_fsdp_root_context"):
                return module._fsdp_root_context
        raise RuntimeError("No _FSDPRootContext found in model.")

    def _start_capturing(self) -> float:
        """Prepare for capture."""
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        self._shared_pool = torch.cuda.graph_pool_handle()
        torch.cuda.set_stream(torch.cuda.Stream())

        gc.freeze()

        logger.info("FSDPCudaGraphHelper: starting CUDA graph capture...")
        return time.time()

    def _finish_capturing(self, start_time: float) -> None:
        """Clean up after capture."""
        gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()

        for module in self.flattened_callables:
            if hasattr(module, "zero_grad_buffer"):
                module.zero_grad_buffer()
        for opt in self.optimizers:
            opt.zero_grad()

        torch.cuda.synchronize()
        torch.distributed.barrier()

        elapsed = time.time() - start_time
        logger.info(
            "FSDPCudaGraphHelper: CUDA graph capture finished in %.2f s.", elapsed
        )
        self._capture_finished = True

    def _build_te_graph_kwargs(self, shims):
        """Build TE ``make_graphed_callables`` kwargs for FP8/MXFP8/NVFP4.

        Detects which modules in *shims* use quantized params and builds
        the per-layer ``fp8_enabled`` tuple and the appropriate recipe.
        """
        def _get_mp_policy(module):
            if hasattr(module, "_mp_policy"):
                return module._mp_policy
            from .mixed_precision import MixedPrecisionPolicy
            return MixedPrecisionPolicy()

        mp_policies = [_get_mp_policy(shim.module) for shim in shims]
        any_fp8 = any(mp.fp8.enabled for mp in mp_policies)
        any_nvfp4 = any(mp.nvfp4.enabled for mp in mp_policies)

        if not any_fp8 and not any_nvfp4:
            return {"fp8_enabled": False}

        from megatron.core.fp4_utils import get_fp4_recipe
        from megatron.core.fp8_utils import get_fp8_recipe

        # Per-layer fp8_enabled: True for quantized modules, False for others
        fp8_enabled = tuple(mp.fp8.enabled or mp.nvfp4.enabled for mp in mp_policies)

        if any_fp8:
            recipe = get_fp8_recipe(self.config)
        else:
            recipe = get_fp4_recipe(self.config)

        kwargs = {
            "fp8_enabled": fp8_enabled,
            "fp8_recipe": recipe,
            "fp8_weight_caching": True,
        }

        # Add fp8_group for amax reduction when TP > 1
        if self.pg_collection is not None:
            fp8_group = self._get_amax_reduction_group()
            if fp8_group is not None:
                kwargs["fp8_group"] = fp8_group

        return kwargs

    def _get_amax_reduction_group(self):
        """Get the FP8 amax reduction group from ``ProcessGroupCollection``.

        Matches ``TECudaGraphHelper._get_amax_reduction_group`` pattern.
        """
        if self.pg_collection is None:
            return None
        pgc = self.pg_collection
        if hasattr(pgc, "tp_dp_cp") and pgc.tp_dp_cp is not None:
            return pgc.tp_dp_cp
        if hasattr(pgc, "tp_cp") and pgc.tp_cp is not None:
            return pgc.tp_cp
        if hasattr(pgc, "tp") and pgc.tp is not None:
            return pgc.tp
        return None

    def create_cudagraphs(self) -> None:
        """Capture CUDA graphs for all discovered modules in one call.

        Prerequisites:
        * ``start_trace()`` has been called.
        * One full forward‑backward step has executed (the trace).
        * ``stop_trace()`` has been called.
        * The allocator is in the *optimized* phase.

        Workflow:
        1. Build shims + sample args from trace records.
        2. Unshard parameters + pop hooks + unshard main‑grad buffers.
        3. Call ``torch.cuda.make_graphed_callables`` with shared pool
           (or TE's ``make_graphed_callables`` if FP8/MXFP8/NVFP4 is active).
        4. Distribute graphs + install patched forwards.
        5. Restore hooks + reshard buffers + reshard parameters.
        """
        assert self._phase == "trace", (
            f"create_cudagraphs called in phase '{self._phase}' "
            f"— call stop_trace() first."
        )
        self._phase = "capture"

        if not self.flattened_callables:
            logger.warning(
                "FSDPCudaGraphHelper: no graphable modules — skipping capture."
            )
            self._capture_finished = True
            return

        start_time = self._start_capturing()

        # ---- 1. Build shims + sample args -------------------------
        shims, sample_args = self._build_from_trace()

        # ---- 2. Unshard parameters --------------------------------
        for module in self.flattened_callables:
            module.unshard()

        # ---- 3. Pop hooks -----------------------------------------
        saved_hooks: List[Tuple[torch.nn.Module, Dict[str, Any]]] = []
        for module in self.flattened_callables:
            saved_hooks.extend(_pop_hooks_recursive(module))

        # ---- 4. Unshard main‑grad buffers -------------------------
        for module in self.flattened_callables:
            for group in module._fsdp_param_groups:
                if hasattr(group, "main_grad_buffer"):
                    group.main_grad_buffer.fetch_buffer()

        # ---- 5. Call make_graphed_callables -----------------------
        ctx = self._get_root_context()
        ctx.cuda_graph_active = True
        try:
            torch.cuda.synchronize()

            te_kwargs = self._build_te_graph_kwargs(shims)
            if HAVE_TE_GRAPHS and te_kwargs.get("fp8_enabled", False):
                graphs = te_make_graphed_callables(
                    tuple(shims),
                    tuple(sample_args),
                    num_warmup_iters=3,
                    allow_unused_input=True,
                    pool=self._shared_pool,
                    **te_kwargs,
                )
            else:
                graphs = torch.cuda.make_graphed_callables(
                    tuple(shims),
                    tuple(sample_args),
                    num_warmup_iters=3,
                    allow_unused_input=True,
                    pool=self._shared_pool,
                )
        finally:
            ctx.cuda_graph_active = False

        # ---- 6. Distribute graphs --------------------------------
        for idx, module in enumerate(self.flattened_callables):
            module._fsdp_cuda_graphs = []
            for mb_idx in range(self.num_microbatches):
                graph_idx = idx * self.num_microbatches + mb_idx
                module._fsdp_cuda_graphs.append(graphs[graph_idx])

            # Store the capture shim — its _none_mask was populated
            # during warmup and is needed by restore_none_positions.
            capture_shim_idx = idx * self.num_microbatches
            module._fsdp_cuda_graph_shim = shims[capture_shim_idx]

            param_names = _get_forward_param_names(module.__class__)
            module._fsdp_cuda_graph_param_names = list(param_names)
            module._fsdp_cuda_graph_tensor_names = list(
                shims[capture_shim_idx].tensor_param_names
            )

        # ---- 7. Install patched forwards -------------------------
        module_orig_forwards: Dict[int, Any] = {}
        for module in self.flattened_callables:
            module_orig_forwards[id(module)] = module.forward

        def _make_patched(module):
            orig_fwd = module_orig_forwards[id(module)]
            graphs_ref = module._fsdp_cuda_graphs
            shim_ref = module._fsdp_cuda_graph_shim
            tensor_names_ref = module._fsdp_cuda_graph_tensor_names
            param_names_ref = module._fsdp_cuda_graph_param_names

            def _patched_fwd(*args, **kwargs):
                microbatch_idx = getattr(module, "current_microbatch", 0)
                graph = graphs_ref[microbatch_idx % len(graphs_ref)]

                bound: Dict[str, Any] = {}
                for i, val in enumerate(args):
                    if i < len(param_names_ref):
                        bound[param_names_ref[i]] = val
                bound.update(kwargs)
                flat = tuple(bound[n] for n in tensor_names_ref)
                result = graph(*flat)
                return shim_ref.restore_none_positions(result)

            module._fsdp_cuda_graph_orig_forward = orig_fwd
            module.forward = _patched_fwd

        for module in self.flattened_callables:
            _make_patched(module)
            # Sentinel: prevents per‑module FSDPCudaGraphRunner capture.
            module._fsdp_cg_runner = True

        # ---- 8. Restore hooks + reshard buffers ------------------
        for submodule, sub_saved in reversed(saved_hooks):
            _restore_hooks(submodule, sub_saved)

        for module in self.flattened_callables:
            for group in module._fsdp_param_groups:
                if hasattr(group, "main_grad_buffer"):
                    group.release_grad_buffer()

        # ---- 9. Reshard parameters -------------------------------
        for module in self.flattened_callables:
            module.reshard()

        self._graphs_created = True
        self._phase = "replay"
        self._finish_capturing(start_time)

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def delete_cuda_graphs(self) -> None:
        """Delete all captured CUDA graphs and restore original forwards."""
        if not self._graphs_created:
            return

        for module in self.flattened_callables:
            if hasattr(module, "_fsdp_cuda_graph_orig_forward"):
                module.forward = module._fsdp_cuda_graph_orig_forward
                del module._fsdp_cuda_graph_orig_forward

            if hasattr(module, "_fsdp_cuda_graphs"):
                for graph in module._fsdp_cuda_graphs:
                    if hasattr(graph, "reset"):
                        graph.reset()
                del module._fsdp_cuda_graphs

            for attr in (
                "_fsdp_cuda_graph_shim",
                "_fsdp_cuda_graph_tensor_names",
                "_fsdp_cuda_graph_param_names",
                "_fsdp_cg_runner",
            ):
                if hasattr(module, attr):
                    delattr(module, attr)

        self._graphs_created = False
        self._phase = "init"
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("FSDPCudaGraphHelper: all CUDA graphs deleted.")
