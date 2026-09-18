# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import cast

import torch.nn as nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantization import (
    COLWISE,
    ROWWISE,
)


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

    The combined path is reachable only with ``overlap_moe_expert_parallel_comm``,
    and that configuration asserts ``mtp_num_layers in (None, 0, 1)`` at
    ``transformer_config.py`` construction time, so this is 0 or 1 by
    construction. The count is read from the model's MTP block rather than from the
    config so a pipeline stage that does not own the block reports 0.
    """
    mtp = getattr(model, "mtp", None)
    layers = getattr(mtp, "layers", None)
    active = 1 if layers else 0
    if active > 1:
        raise AssertionError(
            "Combined/fine-grained 1F1B only supports one MTP layer "
            "(overlap_moe_expert_parallel_comm asserts mtp_num_layers <= 1), but this "
            f"model chunk holds {len(layers)} MTP layers. The per-parameter gradient "
            "multiplicity enumeration does not model deeper MTP."
        )
    return active


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
    *consuming node* rather than once per iteration. The whole space is tiny, because
    ``overlap_moe_expert_parallel_comm`` requires ``mtp_num_layers <= 1``, so the
    counts are enumerated directly:

    * the embedding weight is consumed by the schedule's ``PreProcessNode`` and, when
      the chunk runs MTP, again by the MTP pre-dispatch node:
      ``megatron/core/models/common/fine_grained_callables.py:74-77`` calls
      ``layer._get_embeddings(..., embedding=node.chunk_state.model.embedding, ...)``
      from ``submodule_mtp_pre_dispatch_forward``, i.e. the MTP node explicitly uses
      the model's embedding. That is the recorded desync, so it is 2 with MTP and 1
      without.
    * a tied embedding/output weight is the *same object*, so it also picks up the
      chunk's ``PostProcessNode`` output projection -- one more consumer. It is 3
      with MTP and 2 without, which keeps the no-MTP chunk exactly as it is today
      instead of declaring a consumer that is not there.
    * every other parameter -- including an untied output weight, which its own
      output projection consumes exactly once -- is consumed by one node of this
      unit.

    Because this is an enumeration, it is deliberately not treated as exhaustive:
    a parameter that this function declares too low over-fires loudly, and one it
    declares too high under-fires loudly at the close edge.

    Returns:
        Parameter index -> expected number of gradient contributions in one window.
        An index that is absent is expected once.
    """
    multiplicity: dict[int, int] = {}
    for index, fsdp_parameter in enumerate(unit._trainable_fsdp_parameters()):
        # PreProcessNode's embedding lookup, plus one per MTP pre-dispatch node.
        consumers = 1 + mtp_depth
        if fsdp_parameter.unsharded is shared_weight:
            # The same weight object is the output projection's weight, so it also
            # picks up the PostProcessNode projection. That projection runs with the
            # MTP loss heads when there are any, and its consumer set does not change
            # with MTP depth one way or the other, so this adds exactly one.
            consumers += 1
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
