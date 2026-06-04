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

import torch
import torch.nn as nn


class FSDPCudaGraphRunner:
    """Captures and replays the forward pass of one FSDP module.

    During capture (stage 2), the runner warms up the forward once, then
    records it inside ``torch.cuda.graph()``.  Warmup settles FP8 scale
    factors, RNG state, and cuDNN auto-tuning.

    During replay (stage 3), ``fwd_graph.replay()`` replays the captured
    CUDA kernels directly — no Python FSDP hooks fire.  Non-graphed
    modules continue the normal eager path.

    Parameters:
        fsdp_module: The ``FSDPModule`` to graph.
        warmup_steps: Number of eager warmup passes before capture
            (default 1 — sufficient when the pool is pre-allocated).
    """

    def __init__(self, fsdp_module, warmup_steps: int = 1):
        self._module = fsdp_module
        self._warmup_steps = warmup_steps
        self.fwd_graph: torch.cuda.CUDAGraph | None = None
        self.mempool = torch.cuda.graph_pool_handle()
        self._captured = False
        self._fwd_outputs = None

    # -- Lifecycle ------------------------------------------------------- #

    def capture_forward(self, *args, **kwargs):
        """Warmup, then capture one forward pass of the owning module.

        FSDP hooks fire during both warmup and capture because we call
        ``nn.Module.__call__`` (not the graph-wrapping MegatronModule
        version).  The ``cuda_graph_active`` flag on the root context
        already suppresses side streams and defers reshard.
        """
        # ---- Warmup ----
        for _ in range(self._warmup_steps):
            out = self._run_module(*args, **kwargs)

        # ---- Capture ----
        self.fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fwd_graph, pool=self.mempool):
            self._fwd_outputs = self._run_module(*args, **kwargs)
        self._captured = True
        return self._fwd_outputs

    def replay(self):
        """Replay the captured forward graph.  No Python hooks fire."""
        if not self._captured:
            raise RuntimeError(
                f"FSDPCudaGraphRunner for {self._module} has not been captured yet"
            )
        self.fwd_graph.replay()
        return self._fwd_outputs

    @property
    def captured(self) -> bool:
        return self._captured

    # -- Internal -------------------------------------------------------- #

    def _run_module(self, *args, **kwargs):
        """Call ``nn.Module.__call__`` so that registered forward pre-hooks
        and forward hooks (including FSDP unshard/reshard) fire inside the
        graph capture region."""
        return super(nn.Module, self._module).__call__(*args, **kwargs)
