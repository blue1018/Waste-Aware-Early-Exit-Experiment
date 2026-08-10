from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class DynamicOutput:
    logits: torch.Tensor
    predictions: torch.Tensor
    exit_indices: torch.Tensor
    confidences: torch.Tensor


class ExitHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.classifier(self.dropout(self.norm(pooled)))


def build_static_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if key == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if key in {"resnet18", "resnet18_final_only"}:
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unknown model: {name}")


class EarlyExitResNet18(nn.Module):
    exit_names = ("exit1", "exit2", "final")

    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.num_classes = num_classes
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.exit1 = ExitHead(128, num_classes, dropout)
        self.exit2 = ExitHead(256, num_classes, dropout)
        self.final_pool = backbone.avgpool
        self.final = nn.Linear(backbone.fc.in_features, num_classes)

    def _to_layer2(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.layer1(self.stem(inputs)))

    def forward_all(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        layer2 = self._to_layer2(inputs)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "exit1": self.exit1(layer2),
            "exit2": self.exit2(layer3),
            "final": self.final(self.final_pool(layer4).flatten(1)),
        }

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_all(inputs)

    @staticmethod
    def _temperature(value: float | torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device).clamp_min(1e-6)

    @staticmethod
    def _threshold_vector(values: Sequence[float] | torch.Tensor, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    def forward_dynamic(
        self,
        inputs: torch.Tensor,
        temperatures: Mapping[str, float | torch.Tensor],
        thresholds: Mapping[str, Sequence[float] | torch.Tensor],
    ) -> DynamicOutput:
        batch_size = inputs.shape[0]
        device = inputs.device
        output_logits = torch.zeros(batch_size, self.num_classes, device=device)
        exit_indices = torch.full((batch_size,), 2, dtype=torch.long, device=device)
        confidences = torch.zeros(batch_size, device=device)

        layer2 = self._to_layer2(inputs)
        logits1 = self.exit1(layer2)
        probabilities1 = torch.softmax(logits1 / self._temperature(temperatures["exit1"], device), dim=1)
        confidence1, prediction1 = probabilities1.max(dim=1)
        threshold1 = self._threshold_vector(thresholds["exit1"], device)[prediction1]
        leave1 = confidence1 >= threshold1
        output_logits[leave1] = logits1[leave1]
        exit_indices[leave1] = 0
        confidences[leave1] = confidence1[leave1]

        active1 = (~leave1).nonzero(as_tuple=False).flatten()
        if active1.numel() > 0:
            layer3 = self.layer3(layer2.index_select(0, active1))
            logits2 = self.exit2(layer3)
            probabilities2 = torch.softmax(logits2 / self._temperature(temperatures["exit2"], device), dim=1)
            confidence2, prediction2 = probabilities2.max(dim=1)
            threshold2 = self._threshold_vector(thresholds["exit2"], device)[prediction2]
            leave2_local = confidence2 >= threshold2
            leave2_global = active1[leave2_local]
            output_logits[leave2_global] = logits2[leave2_local]
            exit_indices[leave2_global] = 1
            confidences[leave2_global] = confidence2[leave2_local]

            active2_local = (~leave2_local).nonzero(as_tuple=False).flatten()
            if active2_local.numel() > 0:
                active2_global = active1[active2_local]
                layer4 = self.layer4(layer3.index_select(0, active2_local))
                final_logits = self.final(self.final_pool(layer4).flatten(1))
                final_probabilities = torch.softmax(
                    final_logits / self._temperature(temperatures["final"], device), dim=1
                )
                final_confidence, _ = final_probabilities.max(dim=1)
                output_logits[active2_global] = final_logits
                confidences[active2_global] = final_confidence

        predictions = output_logits.argmax(dim=1)
        return DynamicOutput(output_logits, predictions, exit_indices, confidences)

