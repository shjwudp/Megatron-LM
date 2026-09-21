# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install the sub-module hooks required by MCore combined 1F1B."""

    def register_hooks(submodule, owner):
        if isinstance(submodule, FsdpModule):
            owner = submodule  # BEFORE registering: an FSDP unit owns itself
        if len(list(submodule.parameters(recurse=False))) > 0:
            submodule.register_forward_pre_hook(
                _make_unshard_forward_hook(owner), prepend=True, with_kwargs=True
            )
            submodule.register_full_backward_pre_hook(_make_unshard_backward_hook(owner))
        for child in submodule.children():
            register_hooks(child, owner)

    assert isinstance(module, FsdpModule), "Owner must be an FsdpModule."
    register_hooks(module, module)

    model = _language_model_of(module)
    tied = model.share_embeddings_and_output_weights and model.pre_process
    mtp_depth = _active_mtp_layers(model)
    embedding_weight = getattr(
        getattr(getattr(model, 'embedding', None), 'word_embeddings', None), 'weight', None
    )
    for submodule in module.modules():
        if not isinstance(submodule, FsdpModule):
            continue
        submodule.set_grad_multiplicity(
            multiplicity=_unit_grad_multiplicity(submodule, mtp_depth, embedding_weight, tied)
        )
        submodule.register_post_backward_hook(_module_post_backward_hook)


def _language_model_of(fsdp_unit):
    """Return the language model whose parameters this FSDP unit owns.

    ``fully_shard`` composes the FsdpModule mixin onto whatever class it wraps
    (``fully_shard.py``: ``type(f"ExperimentalFsdp{cls.__name__}", (FsdpModule, cls), {})``),
    and MCore applies the mixed-precision wrapper *before* FSDP. The top-level FSDP
    unit is therefore an ``ExperimentalFsdpFloat16Module`` -- a ``Float16Module``
    whose ``.module`` holds the real GPTModel -- and the model-level attributes the
    multiplicity contract needs (``pre_process``, ``share_embeddings_and_output_weights``,
    ``mtp_process`` and ``embedding``) live on that inner model. Reading them off the
    mixed-precision wrapper either raises ``AttributeError`` or, worse, silently
    yields nothing and under-declares the embedding's multiplicity.
    """
    model = fsdp_unit
    while not hasattr(model, 'pre_process'):
        inner = getattr(model, 'module', None)
        if inner is None or inner is model:
            raise AssertionError(
                f"cannot find a language model inside FSDP unit {type(fsdp_unit).__name__}: "
                "expected a mixed-precision wrapper around a GPTModel. The gradient "
                "multiplicity needs model.pre_process and model.embedding, and guessing "
                "would produce a wrong per-parameter count."
            )
        model = inner
    return model


def _active_mtp_layers(module) -> int:
    """Return whether THIS pipeline stage runs MTP (0 or 1)."""
    depth = getattr(module.config, 'mtp_num_layers', None) or 0
    if depth == 0:
        return 0
    if not hasattr(module, 'mtp_process'):
        raise AssertionError(
            "config.mtp_num_layers is set but the model exposes no `mtp_process`; "
            "cannot tell whether this pipeline stage runs MTP, and guessing would "
            "produce a wrong gradient multiplicity."
        )
    if not module.mtp_process:
        return 0
    assert depth == 1, (
        "overlap_moe_expert_parallel_comm requires mtp_num_layers <= 1 "
        "(transformer_config.py:3316-3320); per-parameter multiplicity does not "
        f"model deeper MTP (got {depth})."
    )
    return 1


def _unit_grad_multiplicity(unit, mtp_depth: int, embedding_weight, tied: bool) -> dict:
    """Per-parameter backward contribution counts for the combined 1F1B path.

    Nearly every parameter's gradient comes from exactly one schedule node, so the
    default is 1. The embedding is the only parameter with extra consumers:

      * this chunk's PreProcessNode embedding lookup          -> the base 1
      * one MTP pre-dispatch node per MTP layer              -> +mtp_depth
      * the PostProcessNode output projection, but ONLY when the output layer
        runs against this very weight object                 -> +1 if tied
    """
    multiplicity = {}
    for fsdp_parameter in unit._trainable_fsdp_parameters():
        consumers = 1
        if fsdp_parameter.unsharded is embedding_weight:
            consumers += mtp_depth + (1 if tied else 0)
        multiplicity[fsdp_parameter.fqns] = consumers
    return multiplicity
