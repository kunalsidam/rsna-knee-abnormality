"""Baseline model: ImageNet ResNet18 with a 12-way multi-label head."""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models

from .config import TARGET_COLS


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(TARGET_COLS))
    return model
