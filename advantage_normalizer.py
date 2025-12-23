import torch
from typing import Optional


class AdvantageNormalizer:
    """Running mean/std (EMA) for advantage normalization."""

    def __init__(self, momentum: float = 0.98, eps: float = 1e-8, device: Optional[torch.device] = None):
        self.momentum = momentum
        self.eps = eps
        self.device = device
        self.mean: Optional[torch.Tensor] = None
        self.var: Optional[torch.Tensor] = None

    def normalize(self, advantages: torch.Tensor) -> torch.Tensor:
        if self.device is None:
            self.device = advantages.device
        else:
            advantages = advantages.to(self.device)

        cur_mean = advantages.mean()
        cur_var = advantages.var(unbiased=False)

        if self.mean is None or self.var is None:
            self.mean = cur_mean.detach()
            self.var = cur_var.detach()
        else:
            self.mean = self.momentum * self.mean + (1 - self.momentum) * cur_mean.detach()
            self.var = self.momentum * self.var + (1 - self.momentum) * cur_var.detach()
        self.var = self.var.clamp(min=1e-2)

        return (advantages - self.mean) / (self.var.sqrt() + self.eps)
