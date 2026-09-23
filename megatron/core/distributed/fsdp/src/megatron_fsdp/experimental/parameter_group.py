# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Parameter-group runtime state for the minimal Megatron-FSDP path."""

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal
from weakref import ReferenceType, ref

import torch
import torch.distributed._symmetric_memory as symm_mem
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Partial, Replicate
from torch.distributed.tensor.placement_types import Placement

from ..mixed_precision import MixedPrecisionPolicy
from ..utils import HAVE_TE
from .dbuffer import DBuffer
from .layout import GlobalLayout
from .module_utils import copy_parameter_attributes, get_parameter_owner

if HAVE_TE:
    from .quantized_dbuffer import QuantizedDBuffer, clear_payloads, effective_dtype
else:

    class QuantizedDBuffer:
        """Fallback for parameter grouping when Transformer Engine is unavailable."""

    def effective_dtype(tensor: torch.Tensor) -> torch.dtype:
        """Without TE, all parameters use their native storage dtype."""
        return tensor.dtype

    def clear_payloads(tensor: torch.Tensor) -> None:
        """Fallback; only reachable for QuantizedDBuffer groups, which require TE."""


# PORT-NOTE: the payload-orientation vocabulary lives here because dev homes it
# in `experimental/quantization.py`, a dev-only new file whose storage helpers
# (`set_rowwise_payload`, `set_columnwise_payload`, `allocate_quantize_temp`,
# `te_cast_master_weights_to_fp8`) are superseded by main's `QuantizedDBuffer` +
# `effective_dtype` and dropped from this port. This lane may not create new
# files, so the vocabulary sits next to its consumer. LANES-PLUMB and
# LANES-MUON call sites should import these names from
# `megatron_fsdp.experimental.parameter_group` (dev imported them from
# `experimental.quantization`).
#: One payload orientation request for an unshard. Regular (non-FP8) parameter
#: groups ignore the value; MXFP8 groups honour it.
ROWWISE = "rowwise"
COLWISE = "colwise"
BOTH = "both"
PayloadOrientation = Literal["rowwise", "colwise", "both"]


def orientation_directions(orientation: str) -> frozenset[str]:
    """Return the payload directions ``orientation`` materializes."""
    if orientation == BOTH:
        return frozenset((ROWWISE, COLWISE))
    return frozenset((orientation,))


def merge_orientations(directions: Iterable[str], default: str = ROWWISE) -> str:
    """Return the narrowest orientation covering every direction in ``directions``.

    Used to widen a materialization that has to serve more than one unshard of the
    same module -- e.g. a recomputed forward and the backward that consumes it
    with no reshard in between, which needs ``"both"``.
    """
    requested = frozenset(directions)
    if not requested:
        return default
    if requested == frozenset((ROWWISE,)):
        return ROWWISE
    if requested == frozenset((COLWISE,)):
        return COLWISE
    return BOTH


_CONTAINING_PARAMETER_GROUP_ATTR = "_mfsdp_parameter_group"


def get_containing_parameter_group(parameter: nn.Parameter) -> "FsdpParameterGroup | None":
    """Return the FSDP parameter group that owns ``parameter``, if any."""
    # This parameter-owned backedge must be weak; otherwise it forms a reference
    # cycle with the parameter group and delays releasing its CUDA storage.
    parameter_group_ref = getattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, None)
    if parameter_group_ref is None:
        return None
    return parameter_group_ref()


def sync_model_weights_from_main_weights(parameters: Iterable[nn.Parameter]) -> None:
    """Sync MFSDP compute weights for parameter groups represented by ``parameters``.

    Parameters outside the experimental MFSDP path are ignored. A parameter group
    may own multiple parameters, but its compute-weight buffer is synced once.
    """
    seen_parameter_groups = set()
    for parameter in parameters:
        if (parameter_group := get_containing_parameter_group(parameter)) is None:
            continue
        if parameter_group in seen_parameter_groups:
            continue
        seen_parameter_groups.add(parameter_group)
        parameter_group.sync_model_weight_from_main_weight()


@dataclass(frozen=True, eq=False)
class FsdpParameter:
    """One physical parameter and its FSDP runtime representations."""

    # Tied weights register one physical parameter under multiple FQNs, all relative
    # to the containing group's owning_module.
    fqns: tuple[str, ...]
    sharded: nn.Parameter
    unsharded: nn.Parameter


class FsdpParameterGroup:
    """A dtype and requires-grad homogeneous group of FSDP-owned parameters."""

    # FsdpModule owns its parameter groups, so this backedge must be weak to avoid
    # a reference cycle that delays releasing CUDA storage until cyclic GC.
    _owning_module: ReferenceType[nn.Module]
    fsdp_parameters: tuple[FsdpParameter, ...]
    mesh: DeviceMesh
    dtype: torch.dtype
    requires_grad: bool
    main_weight: DBuffer
    model_weight: "DBuffer | QuantizedDBuffer"
    # Optimizer-layout representation of model_weight after an optimizer step.
    post_optimizer_model_weight: "DBuffer | QuantizedDBuffer"
    # sync_model_weight_from_main_weight() updates only this rank's optimizer-layout
    # view; the remaining model_weight slices must be all-gathered before compute.
    _model_weight_is_stale: bool
    # Quantized (MXFP8) groups track staleness per payload orientation instead of
    # for the whole buffer, so one materialization neither moves nor clears the
    # other orientation's staleness.
    _rowwise_is_stale: bool
    _colwise_is_stale: bool
    # Only these payload directions are bound to the unsharded parameter tensors.
    _materialized_directions: frozenset[str] = frozenset()
    main_grad: DBuffer | None
    # Optimizer-layout view into main_grad storage, avoiding a second allocation.
    # This is None exactly when main_grad is None.
    pre_optimizer_main_grad: DBuffer | None
    # zero_grad(set_to_none=False) clears only the current optimizer view. If final
    # reduction created a smaller view (e.g. ZeRO-1 or HFSDP), the remaining main_grad
    # storage is stale and must be cleared before the next accumulation begins.
    _main_grad_is_stale: bool
    _unsharded_model_weight: "DBuffer | QuantizedDBuffer"
    _symm_mem_pool: torch.cuda.MemPool | None
    grad_divisor: int

    def __init__(
        self,
        owning_module: nn.Module,
        fqn_to_parameter: dict[str, nn.Parameter],
        mesh: DeviceMesh,
        model_weight_placements: tuple[Placement, ...],
        main_grad_placements: tuple[Placement, ...],
        main_weight_placements: tuple[Placement, ...],
        mixed_precision_policy: MixedPrecisionPolicy,
        grad_divisor: int = 1,
        use_symmetric_memory: bool = False,
        subgroup_size: int | None = None,
    ) -> None:
        """Create persistent sharded buffers for a group of parameters.

        Args:
            owning_module: Closest FSDP root module that owns this parameter group.
            fqn_to_parameter: Root-module-relative FQNs and their parameters.
            mesh: Parent device mesh containing the data-parallel axes.
            model_weight_placements: Compute-weight buffer placements.
            main_grad_placements: Main-gradient buffer placements.
            main_weight_placements: Main-weight buffer placements.
            mixed_precision_policy: Precision policy for main weights and gradients.
            use_symmetric_memory: Allocate communication staging buffers from PyTorch's
                NCCL symmetric-memory pool.
            grad_divisor: Additional divisor applied on top of the mesh-size
                averaging. See ``fully_shard``.
            subgroup_size: Optional contiguous DP parameter-placement subgroup size.
                The value is already normalized to this parameter group's mesh.
        """
        parameter_to_fqns, self.dtype, self.requires_grad = self._collect_parameter_metadata(
            fqn_to_parameter
        )
        self._owning_module = ref(owning_module)
        self.mesh = mesh
        self.grad_divisor = grad_divisor
        self.subgroup_size = subgroup_size
        parameters = tuple(parameter_to_fqns)

        self._initialize_buffers(
            parameters,
            model_weight_placements,
            main_grad_placements,
            main_weight_placements,
            mixed_precision_policy,
            use_symmetric_memory,
        )
        self.fsdp_parameters = self._build_fsdp_parameters(parameter_to_fqns)

        # _build_fsdp_parameters() creates views into this storage, which requires a valid
        # storage size. Release it only after construction; a later unshard reallocates it.
        self._unsharded_model_weight.release_storage()
        self._switch_to_sharded_parameters()

    @staticmethod
    def _collect_parameter_metadata(
        fqn_to_parameter: dict[str, nn.Parameter],
    ) -> tuple[dict[nn.Parameter, list[str]], torch.dtype, bool]:
        """Group tied parameters and validate their shared metadata."""
        if not fqn_to_parameter:
            raise ValueError("FsdpParameterGroup requires at least one parameter.")
        parameter_to_fqns: dict[nn.Parameter, list[str]] = {}
        for fqn, parameter in fqn_to_parameter.items():
            parameter_to_fqns.setdefault(parameter, []).append(fqn)

        # Python dicts preserve insertion order, so parameter_to_fqns and
        # fsdp_parameters define the same stable DBuffer tensor order.
        first_parameter = next(iter(parameter_to_fqns))
        dtype = effective_dtype(first_parameter)
        requires_grad = first_parameter.requires_grad
        for parameter, fqns in parameter_to_fqns.items():
            if effective_dtype(parameter) != dtype:
                raise ValueError(
                    f"Expected parameter {fqns!r} to have dtype {dtype}, "
                    f"got {effective_dtype(parameter)}."
                )
            if parameter.requires_grad != requires_grad:
                raise ValueError(
                    f"Expected parameter {fqns!r} to have requires_grad={requires_grad}, "
                    f"got {parameter.requires_grad}."
                )
        return parameter_to_fqns, dtype, requires_grad

    def _initialize_buffers(
        self,
        parameters: tuple[nn.Parameter, ...],
        model_weight_placements: tuple[Placement, ...],
        main_grad_placements: tuple[Placement, ...],
        main_weight_placements: tuple[Placement, ...],
        mixed_precision_policy: MixedPrecisionPolicy,
        use_symmetric_memory: bool,
    ) -> None:
        """Allocate weight and gradient buffers in their required dependency order."""
        if use_symmetric_memory and not hasattr(symm_mem, "is_symm_mem_tensor"):
            raise RuntimeError("Symmetric-memory MFSDP requires PyTorch 2.12 or later.")

        # All weight and gradient buffers share the same packing and padding.
        layout = GlobalLayout.build(
            (parameter.shape for parameter in parameters),
            dp_size=self.mesh.size(),
            block_size=32 if self.dtype == torch.uint8 else 1,
            subgroup_size=self.subgroup_size,
        )
        main_weight_dtype = mixed_precision_policy.main_params_dtype or torch.float32

        self.main_weight = DBuffer(
            mesh=self.mesh,
            placements=main_weight_placements,
            layout=layout,
            dtype=main_weight_dtype,
            device=self.mesh.device_type,
            subgroup_size=self.subgroup_size,
        )
        for index, parameter in enumerate(parameters):
            if self.dtype == torch.uint8:
                # Use preserved values to initialize optimizer weights without
                # MXFP8 quantization error, then release TE's extra copy.
                initial_value = parameter.get_high_precision_init_val()
                if initial_value is None:
                    raise ValueError(
                        "MXFP8 parameters require preserved high-precision initialization values. "
                        "Use quantized_model_init(preserve_high_precision_init_val=True)."
                    )
                parameter.clear_high_precision_init_val()
            else:
                initial_value = parameter
            self.main_weight.copy_from(index, initial_value)

        if use_symmetric_memory:
            # PyTorch caches this in C++ and returns early when the backend is already NCCL.
            symm_mem.set_backend("NCCL")
            self._symm_mem_pool = symm_mem.get_mem_pool(self.main_weight.device)
        else:
            self._symm_mem_pool = None

        if main_weight_dtype == self.dtype and main_weight_placements == model_weight_placements:
            self.model_weight = self.main_weight
        else:
            with self._symmetric_memory_context():
                if self.dtype == torch.uint8:
                    self.model_weight = QuantizedDBuffer(
                        self.mesh,
                        model_weight_placements,
                        layout,
                        self.main_weight.device,
                        subgroup_size=self.subgroup_size,
                    )
                else:
                    # Keep the configured compute-weight layout alive for the lifetime of this
                    # parameter group. The optimizer-layout sync buffer below is only a view
                    # into its local storage, so the first ZeRO-1 unshard can all-gather
                    # directly into this allocation.
                    self.model_weight = DBuffer(
                        mesh=self.mesh,
                        placements=model_weight_placements,
                        layout=layout,
                        dtype=self.dtype,
                        device=self.main_weight.device,
                        subgroup_size=self.subgroup_size,
                    )
        self.post_optimizer_model_weight = self.model_weight.view(main_weight_placements)
        self.sync_model_weight_from_main_weight()
        with self._symmetric_memory_context():
            if isinstance(self.model_weight, DBuffer):
                self._unsharded_model_weight = DBuffer(
                    mesh=self.mesh,
                    placements=[Replicate()] * self.mesh.ndim,
                    layout=layout,
                    dtype=self.dtype,
                    device=self.main_weight.device,
                    subgroup_size=self.subgroup_size,
                )
            else:
                self._unsharded_model_weight = QuantizedDBuffer(
                    self.mesh,
                    [Replicate()] * self.mesh.ndim,
                    layout,
                    self.main_weight.device,
                    subgroup_size=self.subgroup_size,
                )

        self.main_grad = None
        self.pre_optimizer_main_grad = None
        self._main_grad_is_stale = False
        if not self.requires_grad:
            return

        grad_dtype = mixed_precision_policy.main_grads_dtype or parameters[0].dtype
        # Keep main_grad persistent for the initial implementation. For micro-batch
        # size 1, this allocation could be delayed until post_backward and then
        # eagerly deallocated right after optimizer.step(), avoiding main_grad
        # storage during forward. That requires a separate lifetime contract with
        # the optimizer, so this version keeps the simpler persistent buffer.
        self.main_grad = DBuffer(
            mesh=self.mesh,
            placements=main_grad_placements,
            layout=layout,
            dtype=grad_dtype,
            device=self.main_weight.device,
            subgroup_size=self.subgroup_size,
        )
        self.pre_optimizer_main_grad = self.main_grad.view(main_weight_placements)

    def _build_fsdp_parameters(
        self, parameter_to_fqns: dict[nn.Parameter, list[str]]
    ) -> tuple[FsdpParameter, ...]:
        """Materialize parameter storage and build its FSDP representations."""
        fsdp_parameters: list[FsdpParameter] = []
        main_grad_dtype = self.main_grad.dtype if self.main_grad is not None else None
        for index, (parameter, fqns) in enumerate(parameter_to_fqns.items()):
            # MXFP8 uses get_tensor() to pad scales for TE GEMM. get_tensor_view()
            # exposes compact unswizzled scales, which TE GEMM does not yet accept:
            # https://github.com/NVIDIA/TransformerEngine/issues/3518
            unsharded_tensor = (
                self._unsharded_model_weight.get_tensor_view(index)
                if isinstance(self._unsharded_model_weight, DBuffer)
                else self._unsharded_model_weight.get_tensor(index)
            )
            if parameter.is_meta:
                # A meta Parameter cannot set .data to a real tensor because their
                # TensorImpl types are incompatible, so swap in a materialized Parameter.
                # Copy model metadata first since swap_tensors() also swaps attributes.
                materialized_parameter = nn.Parameter(
                    unsharded_tensor, requires_grad=parameter.requires_grad
                )
                copy_parameter_attributes(parameter, materialized_parameter)
                torch.utils.swap_tensors(parameter, materialized_parameter)
            else:
                parameter.data = unsharded_tensor
                parameter.grad = None
            # Parameter-owned markers must not retain their FSDP module tree.
            setattr(parameter, _CONTAINING_PARAMETER_GROUP_ATTR, ref(self))

            sharded_parameter = nn.Parameter(
                self.main_weight.get_dtensor(index), requires_grad=parameter.requires_grad
            )
            copy_parameter_attributes(parameter, sharded_parameter)
            if main_grad_dtype:
                sharded_parameter.grad_dtype = main_grad_dtype
            setattr(sharded_parameter, _CONTAINING_PARAMETER_GROUP_ATTR, ref(self))
            fsdp_parameters.append(
                FsdpParameter(fqns=tuple(fqns), sharded=sharded_parameter, unsharded=parameter)
            )
        return tuple(fsdp_parameters)

    def _symmetric_memory_context(self):
        if self._symm_mem_pool is None:
            return nullcontext()
        return torch.cuda.use_mem_pool(self._symm_mem_pool)

    def _set_module_parameter(self, fqns: tuple[str, ...], parameter: nn.Parameter) -> None:
        owning_module = self._owning_module()
        if owning_module is None:
            raise RuntimeError("FSDP parameter group outlived its owning module.")
        for fqn in fqns:
            module, parameter_name = get_parameter_owner(owning_module, fqn)
            module._parameters[parameter_name] = parameter

    def _switch_to_sharded_parameters(self) -> None:
        for fsdp_parameter in self.fsdp_parameters:
            self._set_module_parameter(fsdp_parameter.fqns, fsdp_parameter.sharded)

    def _switch_to_unsharded_parameters(self) -> None:
        for fsdp_parameter in self.fsdp_parameters:
            self._set_module_parameter(fsdp_parameter.fqns, fsdp_parameter.unsharded)

    def sync_model_weight_from_main_weight(self) -> None:
        """Refresh compute weights from optimizer weights."""
        if isinstance(self.post_optimizer_model_weight, DBuffer):
            self.main_weight.cast(
                self.post_optimizer_model_weight.dtype, out=self.post_optimizer_model_weight
            )
            self._model_weight_is_stale = (
                self.post_optimizer_model_weight.placements != self.model_weight.placements
            )
            return
        # PORT-NOTE: dev's `Fp8ParameterGroup` re-quantizes the `post_optimizer_rowwise`
        # and `post_optimizer_colwise` views through TE's
        # `cast_master_weights_to_fp8`; this port re-quantizes through main's
        # `QuantizedDBuffer.quantize_`, which fills both orientation views in one
        # pass (it has no per-direction mode -- see the TODO in
        # quantized_dbuffer.py). Per-direction staleness below then gates which
        # orientation `unshard_parameters` moves back into the model_weight planes.
        self.post_optimizer_model_weight.quantize_(self.main_weight)
        self._update_payload_staleness()

    @property
    def post_optimizer_rowwise(self) -> tuple[DBuffer, DBuffer]:
        """The row-wise (data, scale) optimizer-layout views.

        Dev's `Fp8ParameterGroup.post_optimizer_rowwise` is one DBuffer view;
        on main's 4-plane QuantizedDBuffer it is the row-wise plane pair of
        `post_optimizer_model_weight` (the analogue of `post_optimizer_model_weight`:
        quantization fills the view and unshard redistributes the view back into
        the storage). Quantized groups only.
        """
        return self.post_optimizer_model_weight.rowwise_planes

    @property
    def post_optimizer_colwise(self) -> tuple[DBuffer, DBuffer]:
        """The column-wise (data, scale) optimizer-layout views.

        Dev's `Fp8ParameterGroup.post_optimizer_colwise`; see
        :attr:`post_optimizer_rowwise`. Quantized groups only.
        """
        return self.post_optimizer_model_weight.columnwise_planes

    def _update_payload_staleness(self) -> None:
        """Track per-orientation redistribution staleness after re-quantization.

        Sets both orientations' flags together because QuantizedDBuffer's
        quantize_ refreshes both views in one pass (see
        `sync_model_weight_from_main_weight`); `unshard_parameters` clears them
        per orientation as it moves views back into storage. Quantized groups only.
        """
        self._rowwise_is_stale = any(
            view.placements != plane.placements
            for view, plane in zip(self.post_optimizer_rowwise, self.model_weight.rowwise_planes)
        )
        self._colwise_is_stale = any(
            view.placements != plane.placements
            for view, plane in zip(self.post_optimizer_colwise, self.model_weight.columnwise_planes)
        )

    def _redistribute_payloads_into_storage(self, directions: frozenset[str]) -> None:
        """Move refreshed optimizer-layout views back into parameter-layout storage.

        A pass that materializes a single orientation neither moves nor clears
        the staleness of the other one. Quantized groups only.
        """
        if ROWWISE in directions and self._rowwise_is_stale:
            for view, plane in zip(self.post_optimizer_rowwise, self.model_weight.rowwise_planes):
                view.redistribute(plane.placements, out=plane)
            self._rowwise_is_stale = False
        if COLWISE in directions and self._colwise_is_stale:
            for view, plane in zip(
                self.post_optimizer_colwise, self.model_weight.columnwise_planes
            ):
                view.redistribute(plane.placements, out=plane)
            self._colwise_is_stale = False

    def _gather_payload(
        self, source: tuple[DBuffer, DBuffer], target: tuple[DBuffer, DBuffer]
    ) -> None:
        """All-gather one payload orientation's planes into ``target``.

        Without a changed mesh axis the planes share the same local layout and
        `redistribute` copies locally instead. Quantized groups only.
        """
        for source_plane, target_plane in zip(source, target):
            source_plane.redistribute(target_plane.placements, out=target_plane)

    def unshard_parameters(self, orientation: PayloadOrientation = BOTH) -> None:
        """Install full parameters for local compute.

        Args:
            orientation: Which MXFP8 payload orientations to make available for the
                upcoming compute. Regular groups store their full parameter in one
                buffer and ignore this value; MXFP8 groups gather only what the pass
                needs (row-wise for forward, column-wise for backward).
        """
        # PORT-NOTE: `orientation` defaults to BOTH, dev's documented safe
        # superset for a caller that does not know the pass. Dev's signatures
        # default to "rowwise", but every dev call site passes the value
        # explicitly; BOTH only widens a direct call's materialization.
        if isinstance(self.model_weight, QuantizedDBuffer):
            self._unshard_quantized_parameters(orientation)
            return
        if self._model_weight_is_stale:
            self.post_optimizer_model_weight.redistribute(
                self.model_weight.placements, out=self.model_weight
            )
            self._model_weight_is_stale = False

        if self.model_weight.placements == self._unsharded_model_weight.placements:
            unsharded_model_weight = self.model_weight
        else:
            unsharded_model_weight = self._unsharded_model_weight
            with self._symmetric_memory_context():
                unsharded_model_weight.reallocate_storage()
            # This buffer backs unsharded Parameters whose views may be saved by autograd.
            # Autograd records a tensor's version counter when saving it for backward, and
            # in-place writes like the out= redistribution below increment that counter even
            # under no_grad. Without preserving it, backward can fail with "modified by an
            # inplace operation" even though FSDP only materialized internal storage.
            with torch.autograd._unsafe_preserve_version_counter(
                unsharded_model_weight.local_buffer
            ):
                self.model_weight.redistribute(
                    unsharded_model_weight.placements, out=unsharded_model_weight
                )

        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            fsdp_parameter.unsharded.data = unsharded_model_weight.get_tensor_view(index)
        self._switch_to_unsharded_parameters()

    def _unshard_quantized_parameters(self, orientation: PayloadOrientation) -> None:
        """Install full MXFP8 parameters, gathering only ``orientation``'s payloads.

        Dev's `Fp8ParameterGroup.unshard_parameters` semantics adapted to main's
        4-plane QuantizedDBuffer: per-orientation staleness moves, per-orientation
        gathers, per-orientation payload binding, and quantizer usage kept equal
        to the payloads actually bound (TE's `update_usage` raises rather than
        deriving a direction that has no data).
        """
        directions = orientation_directions(orientation)
        missing = directions - self._materialized_directions
        if ROWWISE in missing:
            self._redistribute_payloads_into_storage(frozenset((ROWWISE,)))
        if COLWISE in missing:
            self._redistribute_payloads_into_storage(frozenset((COLWISE,)))
        materialized = self._materialized_directions | directions
        if missing:
            with self._symmetric_memory_context():
                self._unsharded_model_weight.reallocate_storage()
            preserved_tensors = tuple(
                plane.local_buffer for plane in self._unsharded_model_weight.planes
            )
            # This buffer backs unsharded Parameters whose views may be saved by autograd.
            # Autograd records a tensor's version counter when saving it for backward, and
            # in-place writes like the out= redistribution below increment that counter even
            # under no_grad. Without preserving it, backward can fail with "modified by an
            # inplace operation" even though FSDP only materialized internal storage.
            with torch.autograd._unsafe_preserve_version_counter(preserved_tensors):
                # Gather only the payload planes the pass needs: row-wise
                # (forward GEMM) and/or column-wise (backward GEMM).
                if ROWWISE in missing:
                    self._gather_payload(
                        self.model_weight.rowwise_planes,
                        self._unsharded_model_weight.rowwise_planes,
                    )
                if COLWISE in missing:
                    self._gather_payload(
                        self.model_weight.columnwise_planes,
                        self._unsharded_model_weight.columnwise_planes,
                    )
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            tensor = fsdp_parameter.unsharded
            if missing:
                # Rebinding installs every materialized orientation's payloads onto
                # the stable parameter object (MXFP8Tensor._set_data copies the
                # wrapper's attributes), so references autograd saved keep seeing
                # current bindings when a residency window is widened. Dev binds
                # only the missing orientations in place with
                # set_rowwise_payload/set_columnwise_payload; this covers the whole
                # materialized set with equivalent bindings for the same effect.
                tensor.data = self._unsharded_model_weight.get_tensor(
                    index,
                    rowwise=ROWWISE in materialized,
                    columnwise=COLWISE in materialized,
                )
            quantizer = getattr(tensor, "_quantizer", None)
            if quantizer is not None:
                # TE propagates the tensor's quantizer usage into `update_usage` when
                # it takes a primary fp8 weight as a GEMM operand, and MXFP8
                # `update_usage` raises rather than deriving a direction that has no
                # data. Keep the flags equal to the payloads actually bound.
                quantizer.set_usage(
                    rowwise=ROWWISE in materialized, columnwise=COLWISE in materialized
                )
        self._materialized_directions = materialized
        self._switch_to_unsharded_parameters()

    def reshard_parameters(self) -> None:
        """Install sharded DTensor parameters on the owning modules."""
        self._switch_to_sharded_parameters()

    def release_unsharded_storage(self) -> None:
        """Release this group's full-parameter storage."""
        # This method is shared by the post-forward and post-backward release
        # paths. Post-forward must release storage because autograd may have
        # saved forward views into the unsharded parameters. Post-backward could
        # replace unsharded parameter .data with size-0 empty tensors, instead
        # of releasing storage, because autograd has consumed those saved views.
        # That alternative is not much cleaner, and splitting post-forward and
        # post-backward reshard behavior would make the caller code less clean,
        # so keep the shared storage-release path.
        if isinstance(self._unsharded_model_weight, QuantizedDBuffer):
            # Detach the fp8 tensor payloads and forget the materialized directions;
            # the sharded payloads rest in this group's planes until the next
            # unshard rebinds them. Dev's Fp8ParameterGroup.release_unsharded_storage
            # semantics kept.
            for fsdp_parameter in self.fsdp_parameters:
                clear_payloads(fsdp_parameter.unsharded)
            self._materialized_directions = frozenset()
        self._unsharded_model_weight.release_storage()

    def allocate_partial_grad_buffer(self) -> DBuffer:
        """Allocate the unreduced reduce-scatter input buffer."""
        assert self.main_grad is not None

        grads: list[torch.Tensor] = []
        for fsdp_parameter in self.fsdp_parameters:
            if fsdp_parameter.unsharded.grad is None:
                raise RuntimeError(f"Missing gradient for FSDP parameter {fsdp_parameter.fqns!r}.")
            grads.append(fsdp_parameter.unsharded.grad)
        with self._symmetric_memory_context():
            return DBuffer(
                mesh=self.mesh,
                placements=[Partial("avg")] * self.mesh.ndim,
                layout=self.main_weight.layout,
                dtype=grads[0].dtype,
                device=grads[0].device,
                subgroup_size=self.subgroup_size,
            )

    def copy_gradients_to_partial_buffer(self, partial_grad: DBuffer) -> None:
        """Pack full local gradients into an existing reduce-scatter input buffer."""
        # A future fused-wgrad path can write directly into these buffer views.
        for index, fsdp_parameter in enumerate(self.fsdp_parameters):
            partial_grad.get_tensor_view(index).copy_(fsdp_parameter.unsharded.grad)
            fsdp_parameter.unsharded.grad = None

    def _has_sharded_grads(self) -> bool:
        has_any_grad = False
        has_any_missing_grad = False
        for fsdp_parameter in self.fsdp_parameters:
            if fsdp_parameter.sharded.grad is None:
                has_any_missing_grad = True
            else:
                has_any_grad = True
        if has_any_grad and has_any_missing_grad:
            raise RuntimeError("FSDP sharded gradients must be either all set or all None.")
        return has_any_grad

    def reduce_partial_gradients(self, partial_grad: DBuffer, *, is_last_microbatch: bool) -> None:
        """Reduce a packed partial gradient buffer into sharded parameter gradients.

        For HSDP/HFSDP main_grad rests DP-outer-Partial between microbatches,
        accumulating each backward through the standard zero_grad contract; the
        last microbatch reduces the DP-outer axes, finalizing main_grad to
        main_weight's placements (all-reduce to Replicate for HSDP, reduce-scatter
        to Flat for HFSDP) so ``.grad`` is the fully reduced gradient before
        ``optimizer.step()``. With every axis Flat (plain DP) main_grad already
        rests finalized.
        """
        assert self.main_grad is not None
        assert self.pre_optimizer_main_grad is not None

        # zero_grad(set_to_none=True) clears sharded parameter grads, so this
        # backward can reduce directly into main_grad. zero_grad(set_to_none=False)
        # leaves sharded grads installed, so this backward accumulates into main_grad.
        has_sharded_grads = self._has_sharded_grads()
        if self._main_grad_is_stale:
            # In ZeRO-1 and HFSDP, zero_grad(set_to_none=False) only zeros the smaller
            # optimizer view. Clear the persistent full accumulation buffer before this
            # new step; set_to_none=True needs no clear because out= below overwrites it.
            if has_sharded_grads:
                self.main_grad.local_buffer.zero_()
            self._main_grad_is_stale = False

        if can_reduce_into_main_grad := (
            not has_sharded_grads and partial_grad.dtype == self.main_grad.dtype
        ):
            partial_grad.redistribute(self.main_grad.placements, out=self.main_grad)
            reduced_grad = self.main_grad
        else:
            reduced_grad = partial_grad.redistribute(self.main_grad.placements)

        # Scale this backward's contribution before accumulating it so repeated
        # backwards do not repeatedly scale the running total.
        if self.grad_divisor != 1:
            reduced_grad.local_buffer.div_(self.grad_divisor)

        if reduced_grad is not self.main_grad:
            if has_sharded_grads:
                self.main_grad.local_buffer.add_(reduced_grad.local_buffer)
            else:
                self.main_grad.local_buffer.copy_(reduced_grad.local_buffer)

        def install_sharded_grads(main_grad: DBuffer) -> None:
            for index, fsdp_parameter in enumerate(self.fsdp_parameters):
                fsdp_parameter.sharded.grad = main_grad.get_dtensor(index)

        if is_last_microbatch and self.pre_optimizer_main_grad is not self.main_grad:
            # Finalize the deferred DP-outer reduction (all-reduce for HSDP,
            # reduce-scatter for HFSDP) into the persistent buffer's optimizer-layout
            # view before binding the sharded parameter grads.
            self.main_grad.redistribute(
                self.main_weight.placements, out=self.pre_optimizer_main_grad
            )
            self._main_grad_is_stale = True
            install_sharded_grads(self.pre_optimizer_main_grad)
        else:
            # We could install pre_optimizer_main_grad unconditionally because
            # sharded.grad is only read by the optimizer. However, for consistency and
            # debugging, keep sharded.grad valid even between microbatches.
            install_sharded_grads(self.main_grad)
