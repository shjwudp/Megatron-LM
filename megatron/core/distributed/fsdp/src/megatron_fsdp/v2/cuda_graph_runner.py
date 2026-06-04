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

"""CUDA graph capture / replay for individual FSDP modules."""

import inspect
from collections import OrderedDict
from typing import Tuple, List
from typing import Any, Dict, Optional, Tuple

import torch


def _get_forward_param_names(module: torch.nn.Module) -> List[str]:
    """Return the ordered parameter names of module.forward (excluding 'self')."""
    sig = inspect.signature(module.forward)
    return [
        name for name, p in sig.parameters.items()
        if name != "self"
        and p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    ]


def _flatten_args(param_names: List[str], args: tuple, kwargs: dict) -> Tuple[torch.Tensor, ...]:
    """
    Merge args and kwargs into a single positional tuple
    following the order declared in the forward signature.
    Non-tensor values are skipped (they should be baked in at capture time).
    """
    # Build a full mapping: name -> value
    bound = {}
    for i, val in enumerate(args):
        if i < len(param_names):
            bound[param_names[i]] = val
    bound.update(kwargs)

    # Return only tensor values in signature order
    return tuple(bound[name] for name in param_names if name in bound and isinstance(bound[name], torch.Tensor))


def _get_non_tensor_kwargs(param_names: List[str], args: tuple, kwargs: dict) -> dict:
    """Extract non-tensor kwargs to bake into the shim at capture time."""
    bound = {}
    for i, val in enumerate(args):
        if i < len(param_names):
            bound[param_names[i]] = val
    bound.update(kwargs)

    return {name: val for name, val in bound.items() if not isinstance(val, torch.Tensor)}


class _ForwardShim(torch.nn.Module):
    """
    Wraps module.forward so that:
      - non-tensor kwargs are frozen at capture time
      - tensor inputs are passed positionally in signature order
    """

    def __init__(self, module: torch.nn.Module, tensor_param_names: List[str], frozen_kwargs: dict):
        super().__init__()
        self.module = module
        self.tensor_param_names = tensor_param_names
        self.frozen_kwargs = frozen_kwargs  # non-tensor values baked in

    def forward(self, *flat_tensor_args):
        kwargs = dict(zip(self.tensor_param_names, flat_tensor_args))
        kwargs.update(self.frozen_kwargs)
        return self.module.forward(**kwargs)


def _pop_hooks(module: torch.nn.Module) -> Dict[str, Any]:
    """
    Remove all hooks from *module* (non‑recursive) and return a snapshot
    so they can be restored later.
    """
    saved: Dict[str, Any] = {
        "_forward_pre_hooks": module._forward_pre_hooks,
        "_forward_hooks": module._forward_hooks,
        "_backward_hooks": module._backward_hooks,
        "_state_dict_hooks": module._state_dict_hooks,
        "_load_state_dict_pre_hooks": module._load_state_dict_pre_hooks,
    }
    if hasattr(module, "_backward_pre_hooks"):
        saved["_backward_pre_hooks"] = module._backward_pre_hooks

    # Replace with empty ordered dicts (preserves the attribute type)
    for name, value in saved.items():
        if value is not None:
            setattr(module, name, OrderedDict())

    return saved


def _restore_hooks(module: torch.nn.Module, saved: Dict[str, Any]) -> None:
    """Put the hooks back exactly as they were."""
    for name, value in saved.items():
        if value is not None:
            setattr(module, name, value)


class FSDPCudaGraphRunner:
    """
    Wraps an FSDPModule so that ``module(*args, **kwargs)`` can be served
    from a CUDA graph (forward + backward) while bypassing all of the
    module's hooks on the graphed path.

    Usage
    -----
    >>> runner = FSDPCudaGraphRunner(my_fsdp_module, warmup_steps=3)
    >>> runner.capture_forward(sample_input)   # ← provide a sample with the
    >>>                                        #    expected shape/dtype
    >>> runner.install()                       # patches forward
    >>> output = my_fsdp_module(input_batch)   # now uses the graph, no hooks
    >>> runner.uninstall()                     # restore original behaviour
    """

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        warmup_steps: int = 3,
    ):
        self._module: torch.nn.Module = fsdp_module
        self._warmup_steps: int = warmup_steps

        # Will hold the callable returned by make_graphed_callables
        self._graphed: Optional[torch._CudaGraphCallable] = None

        # Original forward so we can restore it later
        self._orig_fwd: Optional[Any] = None

        # Flag flipped by install()/uninstall()
        self._use_cuda_graph: bool = False

        # Book‑keeping
        self._captured: bool = False

    # ------------------------------------------------------------------
    # 1. Capture (forward + backward) with hooks temporarily removed
    # ------------------------------------------------------------------
    def capture_forward(self, *sample_args, **sample_kwargs) -> None:
        # 1. Introspect the module's forward signature
        param_names = _get_forward_param_names(self._module.__class__)
        if torch.distributed.get_rank() == 0:
            print(f"capture_forward, param_names={param_names}", inspect.signature(self._module.__class__.forward))

        # 2. Separate tensor vs non-tensor inputs
        #    - tensors become dynamic positional inputs to the graph
        #    - non-tensors are frozen into the shim
        bound = {}
        for i, val in enumerate(sample_args):
            if i < len(param_names):
                bound[param_names[i]] = val
        bound.update(sample_kwargs)

        tensor_names = [n for n in param_names if n in bound and isinstance(bound[n], torch.Tensor)]
        frozen_kwargs = {n: v for n, v in bound.items() if not isinstance(v, torch.Tensor)}
        flat_sample = tuple(bound[n] for n in tensor_names)

        # 3. Build shim
        shim = _ForwardShim(self._module, tensor_names, frozen_kwargs)

        # 4. Warmup on side stream
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(self._warmup_steps):
                shim(*flat_sample)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        def debug_capture_stages(shim, sample_args, module):
            """Isolate which stage breaks: forward-only or forward+backward."""
            import torch

            # Ensure grads exist (stable addresses)
            for p in module.parameters():
                if p.requires_grad and p.grad is None:
                    p.grad = torch.zeros_like(p)

            # Stage 0: eager sanity check
            print("[DEBUG] Stage 0: eager forward")
            out = shim(*sample_args)
            print(f"[DEBUG] Stage 0: eager forward OK, out type={type(out)}")

            if isinstance(out, tuple):
                loss = out[0].sum()
            else:
                loss = out.sum()

            print("[DEBUG] Stage 0: eager backward")
            loss.backward()
            print("[DEBUG] Stage 0: eager backward OK")

            # Zero grads
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            torch.cuda.synchronize()

            # Stage 1: forward-only graph
            print("[DEBUG] Stage 1: forward-only graph capture")
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                out = shim(*sample_args)
            print("[DEBUG] Stage 1: forward-only graph OK")

            # Zero grads again
            for p in module.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            torch.cuda.synchronize()

            # Stage 2: forward + backward graph
            print("[DEBUG] Stage 2: forward+backward graph capture")
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                out = shim(*sample_args)
                if isinstance(out, tuple):
                    loss = out[0].sum()
                else:
                    loss = out.sum()
                loss.backward()
            print("[DEBUG] Stage 2: forward+backward graph OK")
            print("[DEBUG] All stages passed!")

        # 5. Remove hooks, capture, restore hooks
        saved_hooks = _pop_hooks(self._module)
        try:
            # self._graphed = torch.cuda.make_graphed_callables(
            #     shim,
            #     sample_args=flat_sample,
            #     num_warmup_iters=0,
            # )
            debug_capture_stages(shim, flat_sample, self._module)
        finally:
            _restore_hooks(self._module, saved_hooks)

        # 6. Save the tensor param names so install() can flatten at runtime
        self._tensor_param_names = tensor_names
        self._frozen_kwargs = frozen_kwargs
        self._captured = True

    # ------------------------------------------------------------------
    # 2. Install / uninstall the patched forward
    # ------------------------------------------------------------------
    def install(self) -> None:
        if not self._captured:
            raise RuntimeError("Call capture_forward() first")
        if self._orig_fwd is not None:
            return

        self._orig_fwd = self._module.forward
        graphed = self._graphed
        param_names = _get_forward_param_names(self._module.__class__)
        tensor_names = self._tensor_param_names

        def _patched_fwd(*args, **kwargs):
            if self._use_cuda_graph:
                # Silently flatten args/kwargs → positional tensor tuple
                bound = {}
                for i, val in enumerate(args):
                    if i < len(param_names):
                        bound[param_names[i]] = val
                bound.update(kwargs)
                flat = tuple(bound[n] for n in tensor_names)
                return graphed(*flat)
            return self._orig_fwd(*args, **kwargs)

        self._module.forward = _patched_fwd
        self._use_cuda_graph = True

    def uninstall(self) -> None:
        """
        Restore the original ``forward`` so that the module behaves
        exactly as before (eager execution with hooks).
        """
        if self._orig_fwd is None:
            return  # nothing to restore
        self._module.forward = self._orig_fwd
        self._orig_fwd = None
        self._use_cuda_graph = False

    # ------------------------------------------------------------------
    # 3. Convenience properties / helpers
    # ------------------------------------------------------------------
    @property
    def captured(self) -> bool:
        """True if ``capture_forward`` has been called successfully."""
        return self._captured

    @property
    def using_cuda_graph(self) -> bool:
        """True if the patch is currently active (install() called)."""
        return self._use_cuda_graph

    def reset(self) -> None:
        """
        Fully reset the runner: uninstall the patch, forget the graph,
        and allow a fresh capture later.
        """
        self.uninstall()
        self._graphed = None
        self._captured = False
