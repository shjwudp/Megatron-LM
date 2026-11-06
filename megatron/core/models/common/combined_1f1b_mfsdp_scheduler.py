# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def get_fsdp_root(model) -> FsdpModule | None:
    """Find the FSDP root through DDP/mixed-precision wrapper layers."""
    while model is not None:
        if isinstance(model, FsdpModule):
            return model
        model = getattr(model, "module", None)
    return None


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


def finish_fsdp_backward(module: FsdpModule) -> None:
    """Release weights and reduce a completed unit's accumulated gradients."""
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

    for submodule in module.modules():
        # VPP registers hooks while the shared construction context is still
        # open, before is_root() is available. The supplied chunk is its root.
        if isinstance(submodule, FsdpModule) and submodule is not module:
            submodule.register_post_backward_hook(finish_fsdp_backward)
    # The schedule finishes the root explicitly after all backward segments.
    # A root embedding can contribute from both preprocessing and MTP, so a
    # parameter-count callback can fire before its final contribution arrives.
