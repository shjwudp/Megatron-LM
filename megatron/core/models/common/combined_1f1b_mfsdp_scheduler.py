# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.parameter_group import (
    get_containing_parameter_group,
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


def finish_fsdp_backward(module: FsdpModule) -> None:
    """Release weights and reduce gradients after all scheduled consumers finish."""
    module.reshard()
    module._reduce_gradient_groups()


def reshard_fsdp_module(module: FsdpModule) -> None:
    """Reshard the FSDP module after fine-grained computation."""
    assert isinstance(module, FsdpModule), "Expected an FsdpModule."
    module.reshard()


def register_combined_1f1b_hooks(module: FsdpModule) -> None:
    """Install demand-gather and evaluation-release hooks for the EP schedule."""

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


def register_fsdp_schedule_hooks(model: nn.Module, layer_plans) -> tuple[FsdpModule, ...]:
    """Reduce each unit after its last scheduled backward use, including delayed wgrad.

    Operations:
        1. Collect FSDP units and parameter owners, including an outer wrapper
           that is absent from ``model.modules()``.
        2. Visit layer plans in forward order and assign each unit once. Since
           backward reverses this order, the first forward occurrence identifies
           the shared unit's last backward use.
        3. Register callbacks that finish nested units before their parents,
           after the layer's backward and any delayed weight gradients complete.
        4. Return unassigned units (e.g. owners of tied embeddings and MTP
           projections) for completion after the chunk's pre-process backward.
    """
    remaining = dict.fromkeys(unit for unit in model.modules() if isinstance(unit, FsdpModule))
    # The schedule may receive GPTModel inside an FSDP-owned Float16Module.
    # Its parameter owners include that outer root, which model.modules() omits.
    for parameter in model.parameters():
        group = get_containing_parameter_group(parameter)
        if group is not None and (owner := group._owning_module()) is not None:
            remaining.setdefault(owner, None)
    if not remaining:
        return ()
    for plan in layer_plans:
        units = tuple(unit for unit in plan.layer.modules() if unit in remaining)
        for unit in units:
            del remaining[unit]

        def finish_layer(_module, units=units):
            for unit in reversed(units):
                finish_fsdp_backward(unit)

        plan.set_fsdp_reshard_hooks(reshard_fsdp_module, finish_layer)
    return tuple(reversed(remaining))
