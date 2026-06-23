from typing import Callable, Optional

import torch

from .optimizer import MegatronOptimizer
from .optimizer_config import OptimizerConfig
from .grad_scaler import MegatronGradScaler


class DTensorOptimizer(MegatronOptimizer):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        init_state_fn: Callable,
    ):
        super().__init__(optimizer, config, init_state_fn)
