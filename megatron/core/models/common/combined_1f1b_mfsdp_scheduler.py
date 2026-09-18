# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import cast

import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantization import (
    COLWISE,
    ROWWISE,
)
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_offset


def _make_unshard_forward_hook(owner: FsdpModule):
    """Forward pre-hook: unshard the owning FSDP module before the submodule forward."""

    def hook(submodule, _args, _kwargs):
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        # A forward GEMM consumes the row-wise MXFP8 payload. If the same
        # materialization also has to serve a backward pass (activation
        # recomputation with no reshard between them), the module widens it.
        owner.unshard(orientation=ROWWISE)

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Backward pre-hook: unshard the owning FSDP module before the submodule backward."""

    def hook(submodule, _grad_output):
        # The backward GEMM consumes the column-wise MXFP8 payload.
        owner.unshard(orientation=COLWISE)

    return hook


def _module_post_backward_hook(module: FsdpModule) -> None:
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def finalize_fsdp_backward(module) -> None:
    """Declare the end of an FSDP unit's backward from the schedule.

    The fine-grained schedule calls this from the unit's last backward node, which
    is its only reliable end-of-backward edge: the backward is one ``run_backward``
    per schedule node, so the number of post-accumulate-grad callbacks is not the
    unit's parameter count.

    Every FSDP unit nested inside the hook target is closed too -- an expert
    parameter group is its own unit inside a MoE layer, and it finishes with the
    layer that contains it. Hook targets that are not FSDP units themselves (for
    example a ``HybridStack``) still have their nested units closed.
    """
    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            submodule.finalize_scheduled_backward()


def _active_mtp_layers(model) -> int:
    """Return how many MTP layers this pipeline stage actually runs.

    A pipeline stage can hold an MTP block without running all of its layers: the
    main model shifts the MTP depth by ``get_mtp_layer_offset`` so the layers it
    expects another stage to run are skipped. Only the layers with an offset inside
    this stage's block execute, and each of them consumes the shared embedding once.

    This mirrors the offset that ``fine_grained_callables.submodule_mtp_pre_dispatch_forward``
    uses to select ``node.chunk_state.mtp_hidden_states[offset]``.
    """
    mtp = getattr(model, "mtp", None)
    layers = getattr(mtp, "layers", None)
    if not layers:
        return 0
    config = getattr(model, "config", None)
    if config is None:
        return len(layers)
    pg_collection = getattr(model, "pg_collection", None)
    pp = getattr(pg_collection, "pp", None)
    pp_rank = pp.rank() if pp is not None else 0
    offset = get_mtp_layer_offset(config, getattr(model, "vp_stage", None), pp_rank=pp_rank)
    return max(0, len(layers) - offset)


def _shared_weight(model):
    """Return the embedding weight that the output layer reuses, or ``None``.

    Identity, not the FQN, is what ties the two consumers together: the output layer
    runs against the embedding's own ``Parameter`` object when the model shares them.
    """
    getter = getattr(model, "shared_embedding_or_output_weight", None)
    return getter() if callable(getter) else None


def _unit_grad_multiplicity(unit: FsdpModule, shared_weight, mtp_depth: int) -> dict[int, int]:
    """Return ``unit``'s per-parameter backward contribution counts for combined 1F1B.

    In the combined/fine-grained 1F1B path the backward is one autograd GraphTask per
    schedule node over detached node inputs, so a parameter contributes once per
    *consuming node* rather than once per iteration. The counts below are derived
    from the plan's node set, not guessed:

    * the shared embedding is consumed by the schedule's ``PreProcessNode`` and again
      by every MTP pre-dispatch node:
      ``megatron/core/models/common/fine_grained_callables.py`` calls
      ``layer._get_embeddings(..., embedding=node.chunk_state.model.embedding, ...)``
      from ``submodule_mtp_pre_dispatch_forward``, i.e. the MTP node explicitly uses
      the model's embedding.
    * a tied embedding/output weight is projected once more by the chunk's post
      process node, and once per MTP head when the MTP loss is computed with the
      shared weight (``process_mtp_loss`` calls ``output_layer`` per MTP depth).
    * every other parameter is consumed by exactly one schedule node of this unit.

    Because this is an enumeration, it is deliberately not treated as exhaustive:
    a parameter that this function declares too low over-fires loudly, and one it
    declares too high under-fires loudly at the close edge.

    Returns:
        Parameter index -> expected number of gradient contributions in one window.
        An index that is absent is expected once.
    """
    # The embedding lookup runs in the schedule's PreProcessNode and again in every
    # MTP pre-dispatch node.
    embedding_consumers = 1 + mtp_depth
    # A tied embedding/output weight is projected by the post process node and by
    # each MTP head.
    shared_weight_consumers = 1 + mtp_depth

    multiplicity: dict[int, int] = {}
    for index, fsdp_parameter in enumerate(unit._trainable_fsdp_parameters()):
        consumers = embedding_consumers
        if fsdp_parameter.unsharded is shared_weight:
            consumers += shared_weight_consumers
        if consumers != 1:
            multiplicity[index] = consumers
    return multiplicity


def _register_fsdp_hooks(submodule, owner: FsdpModule):
    """Install the unshard hooks on ``submodule`` and recurse into its children."""
    if isinstance(submodule, FsdpModule):
        owner = submodule  # BEFORE registering: an FSDP unit owns itself
    if len(list(submodule.parameters(recurse=False))) > 0:
        submodule.register_forward_pre_hook(
            _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
        )
        submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
    for child in submodule.children():
        _register_fsdp_hooks(child, owner)


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."
    _register_fsdp_hooks(module, module)

    # A unit whose parameters outlive one schedule node -- the chunk root that owns
    # the shared embedding and the output weight -- declares its real counts. Any
    # other unit is consumed once per parameter by its own layer edge and needs no
    # declaration.
    shared_weight = _shared_weight(module)
    mtp_depth = _active_mtp_layers(module)
    for submodule in cast(nn.Module, module).modules():
        if not isinstance(submodule, FsdpModule):
            continue
        # This path disables the automatic module hooks (``register_hooks=False`` in
        # the MFSDP v2 adapter), so the schedule owns the backward window: the unit's
        # parameters are re-accounted against a declared multiplicity and the reduce
        # is triggered by ``finalize_fsdp_backward`` at the schedule's
        # end-of-backward edge instead of by a callback count.
        submodule.set_grad_multiplicity(
            _unit_grad_multiplicity(submodule, shared_weight, mtp_depth)
        )
        submodule.register_post_backward_hook(_module_post_backward_hook, grad_multiplicity=True)
