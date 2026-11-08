"""Device and mixed-precision settings for PyTorch training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: torch.device
    amp_enabled: bool

    @property
    def pin_memory(self) -> bool:
        return self.device.type == "cuda"

    def autocast(self) -> torch.autocast:
        """Use float16 autocast on CUDA when AMP is enabled."""
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp_enabled)

    def make_grad_scaler(self) -> torch.amp.GradScaler:
        """Create one scaler to reuse throughout a training run."""
        return torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def move_batch(self, batch: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = batch
        return (
            images.to(self.device, non_blocking=self.pin_memory),
            labels.to(self.device, non_blocking=self.pin_memory),
        )


def select_device(*, prefer_cuda: bool = True, enable_amp: bool = True) -> DeviceConfig:
    """Choose the first visible CUDA device, with CPU as fallback."""
    use_cuda = prefer_cuda and torch.cuda.is_available()
    return DeviceConfig(torch.device("cuda:0" if use_cuda else "cpu"), use_cuda and enable_amp)
