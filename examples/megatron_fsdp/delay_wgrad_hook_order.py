# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Show why delayed wgrad needs a callback distinct from PyTorch's grad hook.

This uses a mock delayed linear so it requires only PyTorch:

    python examples/megatron_fsdp/delay_wgrad_hook_order.py
"""

from collections.abc import Callable

import torch


class _DelayedLinearFunction(torch.autograd.Function):
    """Compute dgrad now, but save the real wgrad for ``backward_dw()``."""

    @staticmethod
    def forward(ctx, input_tensor, weight, module):
        ctx.save_for_backward(input_tensor, weight)
        ctx.module = module
        return input_tensor @ weight.t()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        ctx.module.pending_wgrad_inputs = (grad_output.detach(), input_tensor.detach())

        # A zero placeholder makes the parameter's AccumulateGrad node run, so
        # post_accumulate_grad_hook fires, but the real wgrad is still pending.
        grad_input = grad_output @ weight
        placeholder_wgrad = torch.zeros_like(weight)
        return grad_input, placeholder_wgrad, None


class MockDelayedLinear(torch.nn.Module):
    """Small model of the delayed-wgrad callback protocol."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.pending_wgrad_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.wgrad_hooks: list[Callable[[], None]] = []

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return _DelayedLinearFunction.apply(input_tensor, self.weight, self)

    def register_wgrad_accumulation_and_reduce_hooks(self, hook: Callable[[], None]) -> None:
        self.wgrad_hooks.append(hook)

    def backward_dw(self) -> None:
        assert self.pending_wgrad_inputs is not None
        grad_output, input_tensor = self.pending_wgrad_inputs
        self.weight.grad = grad_output.t() @ input_tensor
        self.pending_wgrad_inputs = None
        for hook in self.wgrad_hooks:
            hook()


def grad_state(parameter: torch.nn.Parameter) -> str:
    """Return a compact description of the parameter gradient."""
    if parameter.grad is None:
        return "None"
    nonzero = torch.count_nonzero(parameter.grad).item()
    return "zero placeholder" if nonzero == 0 else f"materialized (nonzero={nonzero})"


def main() -> None:
    """Run activation backward first, then the delayed weight-gradient pass."""
    linear = MockDelayedLinear(hidden_size=16)
    weight = linear.weight
    phase = "setup"

    def pytorch_post_accumulate_hook(parameter: torch.nn.Parameter) -> None:
        print(f"PyTorch hook during {phase}: weight.grad={grad_state(parameter)}")

    def delayed_wgrad_hook() -> None:
        print(f"Delayed-wgrad hook during {phase}: weight.grad={grad_state(weight)}")

    weight.register_post_accumulate_grad_hook(pytorch_post_accumulate_hook)
    linear.register_wgrad_accumulation_and_reduce_hooks(delayed_wgrad_hook)

    input_tensor = torch.randn(8, 16, requires_grad=True)
    loss = linear(input_tensor).float().square().sum()

    phase = "backward()"
    loss.backward()
    print(f"After backward(): weight.grad={grad_state(weight)}")
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0

    phase = "backward_dw()"
    linear.backward_dw()
    print(f"After backward_dw(): weight.grad={grad_state(weight)}")
    assert weight.grad is not None and torch.count_nonzero(weight.grad) > 0


if __name__ == "__main__":
    main()
