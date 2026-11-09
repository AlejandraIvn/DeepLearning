"""ImageNet initialization and frozen-backbone transfer learning for PAPILA."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import Normalization


RESNET18 = "resnet18"
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def imagenet_normalization() -> Normalization:
    transform = WEIGHTS.transforms()
    return Normalization(tuple(transform.mean), tuple(transform.std))


class PapilaResNet18(nn.Module):
    """A frozen ImageNet ResNet18 with a trainable one-logit linear head."""

    def __init__(self, *, pretrained: bool) -> None:
        super().__init__()
        self.backbone = resnet18(weights=WEIGHTS if pretrained else None, progress=False)
        if pretrained:
            path = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != WEIGHTS_SHA256:
                    raise ValueError(f"SHA-256 inesperado para los pesos oficiales: {path}")
        self.backbone.requires_grad_(False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
        self.train()

    @property
    def fc_out(self) -> nn.Linear:
        return self.backbone.fc

    def train(self, mode: bool = True) -> PapilaResNet18:
        super().train(mode)
        # Frozen BatchNorm must keep the ImageNet statistics during head training.
        self.backbone.eval()
        self.fc_out.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images).squeeze(1)
