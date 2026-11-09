# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Whole-matrix Muon kernels shared by TP=1 DDP and MFSDP owner updates."""

from dataclasses import dataclass

import torch

# r2s10 schedule used by the internal distributed Muon implementation. Keep the
# coefficients local: changing EO's process-wide registry also changes other optimizers.
R2S10 = (
    (4.732508849, -12.335020239, 8.796253262),
    (4.471004450, -7.504230597, 3.612792015),
    (3.545602705, -4.209203492, 1.447531864),
    *((3.640357367, -4.583299393, 1.733817117),) * 4,
    (3.087466682, -3.887196433, 1.470488208),
    (2.379997469, -2.344821481, 0.926746169),
    (2.0, -1.5, 0.5),
)


def tag_muon_parameters(model_chunks):
    """Describe logical splits before optimizer master parameters are created.

    MLA's main Q up-projection is not a fused QKV weight. Only the indexer's Q
    projection gets the index-head rule. Grouped expert FC1 weights remain 2-D
    individual parameters (weight0, weight1, ...).
    """
    for chunk in model_chunks:
        cfg = chunk.config
        for name, param in chunk.named_parameters():
            if not param.requires_grad or param.ndim != 2:
                continue
            if "word_embeddings.weight" in name:
                # Includes the MTP embedding replica, even on stages without the input embedding.
                param.is_embedding_or_output_parameter = True
            if "linear_fc1.weight" in name:
                param.muon_split_kind = "fc1"
            elif "eh_proj.weight" in name:
                param.muon_split_kind = "eh"
            elif "linear_qkv.weight" in name:
                param.muon_split_kind = "qkv"
                param.muon_head_dim = cfg.kv_channels
            elif "index_linear_q.weight" in name or (
                ".indexer." in name and name.endswith(".linear_wq_b.weight")
            ):
                param.muon_split_kind = "index_q"
                param.muon_head_dim = cfg.dsa_indexer_head_dim


@dataclass(frozen=True)
class MuonMatrixOrthogonalizer:
    """Orthogonalize complete matrices; communication and momentum belong to the caller."""

    coefficient_type: str = "quintic"
    num_ns_steps: int = 5
    scale_mode: str = "spectral"
    extra_scale_factor: float = 1.0
    ns_eps: float | None = None
    split_qkv: bool = False
    split_head: bool = False
    split_fc1: bool = False
    split_eh: bool = False
    qkv_split_shapes: tuple[int, ...] | None = None
    use_syrk: bool = False

    def __post_init__(self):
        if self.num_ns_steps < 1:
            raise ValueError("Muon requires at least one NS step")
        if self.ns_eps is not None and self.ns_eps <= 0:
            raise ValueError("Muon NS epsilon must be positive")

    def _matrix(self, grad):
        from emerging_optimizers.orthogonalized_optimizers import get_muon_scale_factor
        from emerging_optimizers.orthogonalized_optimizers.muon_utils import newton_schulz

        kwargs = {"use_syrk": self.use_syrk}
        if self.ns_eps is not None:
            kwargs["eps"] = self.ns_eps
        coefficient_type = self.coefficient_type
        if coefficient_type == "r2s10":
            coefficient_type = "custom"
            # EO cycles custom sets. Expand to the requested length to implement repeat-last.
            kwargs["custom_coefficient_sets"] = [
                R2S10[min(i, len(R2S10) - 1)] for i in range(self.num_ns_steps)
            ]
        result = newton_schulz(
            grad, steps=self.num_ns_steps, coefficient_type=coefficient_type, **kwargs
        )
        scale = get_muon_scale_factor(*grad.shape, mode=self.scale_mode)
        return result * (scale * self.extra_scale_factor)

    def __call__(self, param, grad):
        kind = getattr(param, "muon_split_kind", None)
        if kind is None and getattr(param, "is_qkv", False):
            kind = "qkv"
        if kind == "qkv" and self.split_qkv:
            shapes = getattr(param, "qkv_split_shapes", None) or self.qkv_split_shapes
            if not shapes or grad.shape[0] % sum(shapes):
                raise ValueError("Muon QKV split requires compatible qkv_split_shapes")
            grouped = grad.reshape(-1, sum(shapes), grad.shape[1])
            pieces = grouped.split(shapes, dim=1)
            outputs = [self._matrix(x.reshape(-1, grad.shape[1])).view_as(x) for x in pieces]
            return torch.cat(outputs, dim=1).reshape_as(grad)
        if kind in ("qkv", "index_q") and self.split_head:
            head_dim = getattr(param, "muon_head_dim", None)
            if not head_dim or grad.shape[0] % head_dim:
                raise ValueError("Muon head split requires a compatible muon_head_dim")
            return torch.cat([self._matrix(x) for x in grad.split(head_dim, dim=0)], dim=0)
        split_dim = None
        if kind == "fc1" and self.split_fc1:
            split_dim = 0
        elif kind == "eh" and self.split_eh:
            split_dim = 1
        if split_dim is not None:
            if grad.shape[split_dim] % 2:
                raise ValueError(f"Muon {kind} split requires an even dimension")
            return torch.cat(
                [self._matrix(x.contiguous()) for x in grad.chunk(2, dim=split_dim)], dim=split_dim
            )
        return self._matrix(grad)
