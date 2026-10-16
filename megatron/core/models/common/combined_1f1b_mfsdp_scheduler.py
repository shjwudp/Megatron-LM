# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


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

    for submodule in module.modules():
        if isinstance(submodule, FsdpModule):
            submodule.register_post_backward_hook(_module_post_backward_hook)


def prefetch_fsdp_parameters(module, *, backward, compute_stream):
    """Queue the next scheduled node's MFSDP v2 gathers without a compute wait.

    Demand hooks wait on each owner's event inside the node's CUDA stream.
    Parameter release and gradient completion remain with the existing hooks.
    The target subtree and phase are selected by the combined scheduler rather
    than inferred from independent forward/backward construction orders.
    """
    units = []

    def collect(submodule):
        if isinstance(submodule, FsdpModule):
            units.append(submodule)
        children = list(submodule.children())
        for child in reversed(children) if backward else children:
            collect(child)

    collect(module)
    if not units:
        return
    policy = units[0]._schedule_policy
    budget = policy.backward_prefetch_size if backward else policy.forward_prefetch_size
    if budget == 0:
        return
    size = 0
    for unit in units:
        unit.context.allgather_stream.wait_stream(compute_stream)
        unit._unshard_parameter_groups()
        size += unit.num_parameter_elements
        if budget is None or size >= budget:
            break
