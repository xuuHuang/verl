# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import numpy as np


class AlphaScheduler:
    """Base class for alpha schedulers. Schedules the alpha parameter for TAPO advantage estimation.

    The scheduler decays alpha from alpha_max to alpha_min over total_steps.

    Args:
        alpha_max: Maximum alpha value.
        total_steps: Total number of training steps over which to decay.
        alpha_min: Minimum alpha value at the end of scheduling (default 0.0).
    """

    def __init__(self, alpha_max: float, total_steps: int, alpha_min: float = 0.0):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.total_steps = total_steps

    def get_alpha(self, step: int) -> float:
        raise NotImplementedError


class ConstantAlphaScheduler(AlphaScheduler):
    """Constant alpha, always returns alpha_max."""

    def get_alpha(self, step: int) -> float:
        return self.alpha_max


class CosineAlphaScheduler(AlphaScheduler):
    """Cosine decay from alpha_max to alpha_min. No warmup."""

    def get_alpha(self, step: int) -> float:
        progress = min(step / max(self.total_steps, 1), 1.0)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * (1 + np.cos(np.pi * progress)) / 2


class LinearAlphaScheduler(AlphaScheduler):
    """Linear decay from alpha_max to alpha_min. No warmup."""

    def get_alpha(self, step: int) -> float:
        progress = min(step / max(self.total_steps, 1), 1.0)
        return self.alpha_max - (self.alpha_max - self.alpha_min) * progress


class ExponentialAlphaScheduler(AlphaScheduler):
    """Exponential decay from alpha_max to alpha_min. No warmup."""

    def __init__(self, alpha_max: float, total_steps: int, alpha_min: float = 0.0):
        super().__init__(alpha_max, total_steps, alpha_min)
        if alpha_min <= 0:
            self.gamma = 0.001 ** (1.0 / max(total_steps, 1))
        else:
            self.gamma = (alpha_min / alpha_max) ** (1.0 / max(total_steps, 1))

    def get_alpha(self, step: int) -> float:
        progress = min(step, self.total_steps)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * (self.gamma ** progress)


def get_alpha_scheduler(tapo_config, total_steps: int) -> AlphaScheduler:
    """Factory function to create an alpha scheduler based on configuration.

    Args:
        tapo_config: Configuration object with optional alpha_scheduler block.
        total_steps: Total number of training steps.

    Returns:
        AlphaScheduler instance.
    """
    scheduler_cfg = tapo_config.get("alpha_scheduler", None)

    scheduler_type = scheduler_cfg.get("type", "constant")
    alpha_max = scheduler_cfg.get("alpha_max", 1.0)
    alpha_min = scheduler_cfg.get("alpha_min", 0.0)

    if scheduler_type == "constant":
        return ConstantAlphaScheduler(alpha_max=alpha_max, total_steps=total_steps)
    elif scheduler_type == "cosine":
        return CosineAlphaScheduler(alpha_max=alpha_max, total_steps=total_steps, alpha_min=alpha_min)
    elif scheduler_type == "linear":
        return LinearAlphaScheduler(alpha_max=alpha_max, total_steps=total_steps, alpha_min=alpha_min)
    elif scheduler_type == "exponential":
        return ExponentialAlphaScheduler(alpha_max=alpha_max, total_steps=total_steps, alpha_min=alpha_min)
    else:
        raise ValueError(
            f"Unknown alpha scheduler type: {scheduler_type}. "
            "Supported: constant, cosine, linear, exponential."
        )
