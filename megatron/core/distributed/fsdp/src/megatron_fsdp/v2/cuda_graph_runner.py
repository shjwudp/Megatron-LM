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


class FSDPCudaGraphRunner:
    """Captures and replays the compute forward of one FSDP module.

    Only ``module.forward()`` is captured — FSDP hooks (``unshard`` for
    buffer allocation + all-gather, ``reshard`` for release) run eagerly
    outside the graph, managed by ``FSDPModule.__call__``.

    Warmup runs ``module.forward()`` eagerly before capture to settle
    FP8 scale factors, RNG state, and cuDNN auto-tuning.  The first
    eager call via ``FSDPModule.__call__`` already ran the hooks and
    forward, so params are unsharded and pool buffers are allocated.

    Parameters:
        fsdp_module: The ``FSDPModule`` to graph.
        warmup_steps: Number of eager forward() passes before capture
            (default 1 — sufficient when pool is pre-allocated).
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
        """Warmup compute, then record it inside ``torch.cuda.graph()``.

        Warmup runs ``module.forward()`` eagerly to settle FP8 scale
        factors, RNG state, and cuDNN auto-tuning.  Then only
        ``module.forward()`` is captured.
        """
        # ---- Warmup ----
        for _ in range(self._warmup_steps):
            self._module.forward(*args, **kwargs)

        # ---- Capture ----
        self.fwd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fwd_graph, pool=self.mempool):
            self._fwd_outputs = self._module.forward(*args, **kwargs)
        self._captured = True
        return self._fwd_outputs

    def replay(self):
        """Replay the captured compute graph."""
        if not self._captured:
            raise RuntimeError(
                f"FSDPCudaGraphRunner for {self._module} has not been captured yet"
            )
        self.fwd_graph.replay()
        return self._fwd_outputs

    @property
    def captured(self) -> bool:
        return self._captured
