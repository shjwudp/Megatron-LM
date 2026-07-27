# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Placement layout and runtime state for Megatron FSDP parameter groups."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import torch

from .buffer_index import Placement
from .dp_buffer import DataParallelBuffer
from .mixed_precision import WeightBufferRole

Placements = tuple[Placement, ...]


@dataclass(frozen=True)
class ParameterGroupLayout:
    """Persistent data-parallel placements used by a parameter group."""

    weight: Placements
    main_weight: Placements
    grad_storage: Placements
    grad_accumulation: Placements

    def validate(self, mesh_ndim: int) -> None:
        """Validate that every placement vector matches the device-mesh rank."""
        for placements in (
            self.weight,
            self.main_weight,
            self.grad_storage,
            self.grad_accumulation,
        ):
            if len(placements) != mesh_ndim:
                raise ValueError(f"Expected {mesh_ndim} placements, got {placements}")

    @classmethod
    def from_strategies(
        cls, sharding_strategy: str, outer_dp_sharding_strategy: str | None = None
    ) -> "ParameterGroupLayout":
        """Resolve public sharding strategies into a placement-only layout."""
        valid_inner = ("no_shard", "optim", "optim_grads", "optim_grads_params")
        if sharding_strategy not in valid_inner:
            raise ValueError(f"Unsupported sharding strategy: {sharding_strategy}")

        weight = (
            Placement.SHARD if sharding_strategy == "optim_grads_params" else Placement.REPLICATE
        )
        optimizer = Placement.REPLICATE if sharding_strategy == "no_shard" else Placement.SHARD
        reduce_each_microbatch = sharding_strategy in ("optim_grads", "optim_grads_params")
        grad_accumulation = Placement.SHARD if reduce_each_microbatch else Placement.PARTIAL
        grad_storage = Placement.SHARD if reduce_each_microbatch else Placement.REPLICATE
        inner_layout = cls(
            weight=(weight,),
            main_weight=(optimizer,),
            grad_storage=(grad_storage,),
            grad_accumulation=(grad_accumulation,),
        )
        if outer_dp_sharding_strategy is None:
            return inner_layout

        if outer_dp_sharding_strategy not in ("no_shard", "optim"):
            raise ValueError(
                f"Unsupported outer DP sharding strategy: {outer_dp_sharding_strategy}"
            )
        if outer_dp_sharding_strategy == "optim" and sharding_strategy != "optim_grads_params":
            raise NotImplementedError(
                "Outer-DP optimizer sharding requires inner optim_grads_params, "
                f"got {sharding_strategy}"
            )
        outer_optimizer = (
            Placement.SHARD if outer_dp_sharding_strategy == "optim" else Placement.REPLICATE
        )
        return cls(
            weight=(Placement.REPLICATE, inner_layout.weight[0]),
            main_weight=(outer_optimizer, inner_layout.main_weight[0]),
            grad_storage=(Placement.REPLICATE, inner_layout.grad_storage[0]),
            grad_accumulation=(Placement.PARTIAL, inner_layout.grad_accumulation[0]),
        )

    @classmethod
    def fsdp(cls) -> "ParameterGroupLayout":
        """Build a one-dimensional fully sharded layout."""
        return cls.from_strategies("optim_grads_params")

    @classmethod
    def hsdp(cls, *, shard_optimizer_across_outer_dp: bool) -> "ParameterGroupLayout":
        """Build the two-dimensional HSDP layout discussed in the design document."""
        return cls.from_strategies(
            "optim_grads_params",
            outer_dp_sharding_strategy=("optim" if shard_optimizer_across_outer_dp else "no_shard"),
        )


class GradientPhase(enum.Enum):
    """Lifecycle phase of the value stored in persistent gradient storage."""

    EMPTY = enum.auto()
    ACCUMULATING = enum.auto()
    READY = enum.auto()


@dataclass(frozen=True)
class PendingWeightTransition:
    """An asynchronously produced persistent weight placement."""

    target: Placements
    event: torch.cuda.Event


@dataclass
class WeightRepresentationState:
    """Runtime state for one model-weight representation."""

    persistent: DataParallelBuffer
    valid_placements: Placements
    full: DataParallelBuffer | None = None
    pending: PendingWeightTransition | None = None


@dataclass
class GradientState:
    """Runtime state for gradient storage and temporary communication buffers."""

    persistent: DataParallelBuffer
    phase: GradientPhase = GradientPhase.EMPTY
    full: DataParallelBuffer | None = None
    communication: DataParallelBuffer | None = None


class ParameterGroupStateView:
    """Read-only compatibility view over independent weight and gradient state."""

    def __init__(
        self, weights: dict[WeightBufferRole, WeightRepresentationState], gradient: GradientState
    ) -> None:
        self._weights = weights
        self._gradient = gradient

    @property
    def weight_valid_by_role(self) -> dict[WeightBufferRole, Placements]:
        """Return valid placements for every weight representation."""
        return {role: state.valid_placements for role, state in self._weights.items()}

    @property
    def full_weights(self) -> dict[WeightBufferRole, DataParallelBuffer]:
        """Return active full-weight leases."""
        return {role: state.full for role, state in self._weights.items() if state.full is not None}

    @property
    def pending_weights(self) -> dict[WeightBufferRole, PendingWeightTransition]:
        """Return pending persistent-weight transitions."""
        return {
            role: state.pending
            for role, state in self._weights.items()
            if state.pending is not None
        }

    @property
    def weight_valid(self) -> Placements:
        """Return valid placements for the canonical model weight."""
        return self._weights[WeightBufferRole.MODEL].valid_placements

    @property
    def full_weight(self) -> DataParallelBuffer | None:
        """Return the canonical full model-weight lease."""
        return self._weights[WeightBufferRole.MODEL].full

    @property
    def grad_phase(self) -> GradientPhase:
        """Return the gradient lifecycle phase."""
        return self._gradient.phase

    @property
    def full_grad(self) -> DataParallelBuffer | None:
        """Return the full-gradient view or temporary lease."""
        return self._gradient.full

    @property
    def grad_comm(self) -> DataParallelBuffer | None:
        """Return the gradient communication workspace."""
        return self._gradient.communication
