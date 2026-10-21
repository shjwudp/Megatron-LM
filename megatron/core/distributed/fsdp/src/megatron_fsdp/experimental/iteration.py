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

"""Capture one FSDP iteration and replay its demand occurrences."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal
from weakref import ref

from .execution_order import ExecutionOrder

if TYPE_CHECKING:
    from .context import FsdpContext
    from .module import FsdpModule

DemandPhase = Literal["forward", "backward", "recompute", "none"]
Demand = tuple[ref["FsdpModule"], DemandPhase]


class IterationOrder(ExecutionOrder[Demand]):
    """One captured order with separate execution and prefetch progress.

    State lives on FsdpContext because autograd hooks can run on another thread.
    The scheduler owns iteration boundaries and must explicitly recapture when
    the training schedule changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.captured = False
        self.mode: Literal["static", "capture", "replay", "demand"] = "static"
        self._prefetch_position = 0

    @property
    def active(self) -> bool:
        return self.mode != "static"

    @property
    def recording(self) -> bool:
        return self.mode == "capture"

    def reset(self) -> None:
        super().reset()
        self._prefetch_position = 0

    def prefetch(self, budget: int | None) -> None:
        """Prefetch occurrences without advancing the execution cursor.

        Targets depend on order and budget, not materialization state. Demand
        still rechecks materialization after any intervening parameter release.
        """
        if self.mode != "replay" or budget == 0:
            return
        self._prefetch_position = max(self._prefetch_position, self.position)
        size = 0
        while True:
            demand = self.next_item(self._prefetch_position - self.position + 1)
            if demand is None:
                break
            module = demand[0]()
            if module is None:
                raise RuntimeError("A module in the captured FSDP iteration was destroyed.")
            self._prefetch_position += 1
            module._unshard_parameter_groups()
            size += module.num_parameter_elements
            if budget is None or size >= budget:
                break


@contextmanager
def capture(context: "FsdpContext") -> Iterator[IterationOrder]:
    """Replace the order by capturing one complete, repeatable training iteration.

    Capture disables speculative prefetch. The scheduler resets the resulting
    order before each replay; forward-only schedules must bypass this F/B order.
    The caller still owns finish_grad_sync after all backward producers.
    """
    context.ensure_finalized()
    if context.iteration_order.active:
        raise RuntimeError("Cannot capture inside an active FSDP iteration.")
    order = context.iteration_order = IterationOrder()
    order.mode = "capture"
    try:
        context.allgather_stream.wait_stream(context.current_stream())
        yield order
        order.captured = True
    finally:
        order.mode = "static"
