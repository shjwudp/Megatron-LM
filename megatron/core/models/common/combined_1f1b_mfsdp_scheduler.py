# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import (
    FsdpModule,
    _collect_owned_parameters,
)


def _make_unshard_forward_hook(owner: FsdpModule):
    """Create a forward pre-hook that unshards the owning FSDP module before the submodule
    forward."""

    def hook(submodule, _args, _kwargs):
        if owner.is_root():
            context = owner.context
            context.allgather_stream.wait_stream(context.current_stream())
        owner.unshard()

    return hook


def _make_unshard_backward_hook(owner: FsdpModule):
    """Create a backward pre-hook that unshards the owning FSDP module before the submodule
    backward."""

    def hook(submodule, _grad_output):
        owner.unshard()

    return hook


def _module_post_forward_hook(module: FsdpModule, _args, _output) -> None:
    """Reshard after evaluation, which bypasses the combined schedule's release hooks."""
    if not module.training:
        module.reshard()


def _module_post_backward_hook(module: FsdpModule) -> None:
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install execution hooks for units whose callback counts are set at construction."""

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

    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            submodule.register_forward_hook(_module_post_forward_hook)
            submodule.register_post_backward_hook(_module_post_backward_hook)


def get_ep_overlap_grad_accumulation_count(fsdp_unit: nn.Module, model: nn.Module) -> int | None:
    """Count owned trainable parameters plus extra callbacks for the embedding."""
    config = getattr(model, "config", None)
    if config is None or not config.overlap_moe_expert_parallel_comm:
        return None

    # Count this chunk's embedding consumers. LM and MTP output heads
    # share one post-process GraphTask, so together they contribute once.
    embedding_consumers = (
        int(model.pre_process)
        + model.get_mtp_depth()
        + int(model.post_process and model.share_embeddings_and_output_weights)
    )
    embedding_extra_callbacks = max(embedding_consumers - 1, 0)

    word_embeddings = getattr(getattr(model, "embedding", None), "word_embeddings", None)
    embedding_weight = getattr(word_embeddings, "weight", None)

    parameters = {
        id(parameter)
        for parameter in _collect_owned_parameters(fsdp_unit).values()
        if parameter.requires_grad
    }
    return len(parameters) + (
        embedding_extra_callbacks if id(embedding_weight) in parameters else 0
    )
