# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TE-compatible CUDA graph callable runtime.

Vendored from https://github.com/buptzyb/te-graph-runtime
Prefer ``pip install te-graph-runtime`` instead of the vendored copy.
"""

from .graph import (
    UPSTREAM_TE_COMMIT,
    UPSTREAM_TE_GRAPH_PATH,
    UPSTREAM_TE_VERSION,
    cuda_graph_checkpoint_context_fn,
    cuda_graph_checkpoint_phase,
    current_cuda_graph_checkpoint_region,
    make_graphed_callables,
    resolve_replay_phase,
    wrap_cuda_graph_checkpoint,
)

__all__ = [
    "UPSTREAM_TE_COMMIT",
    "UPSTREAM_TE_GRAPH_PATH",
    "UPSTREAM_TE_VERSION",
    "cuda_graph_checkpoint_context_fn",
    "cuda_graph_checkpoint_phase",
    "current_cuda_graph_checkpoint_region",
    "make_graphed_callables",
    "resolve_replay_phase",
    "wrap_cuda_graph_checkpoint",
]
