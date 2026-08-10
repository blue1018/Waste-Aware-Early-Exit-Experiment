from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossBreakdown:
    total: torch.Tensor
    ce: torch.Tensor
    kd: torch.Tensor
    per_exit_ce: dict[str, torch.Tensor]


def compute_class_weights(
    counts: torch.Tensor,
    minimum: float = 0.5,
    maximum: float = 3.0,
) -> torch.Tensor:
    values = counts.to(dtype=torch.float32)
    if torch.any(values <= 0):
        raise ValueError("Class counts must be positive")
    raw = torch.sqrt(values.sum() / (len(values) * values))
    normalized = raw / raw.mean()
    return normalized.clamp(min=minimum, max=maximum)


def multi_exit_loss(
    logits: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    class_weights: torch.Tensor | None,
    alpha: Sequence[float],
    kd_temperature: float,
    kd_gamma: float,
    use_distillation: bool,
) -> LossBreakdown:
    names = ("exit1", "exit2", "final")
    if set(names) - set(logits):
        raise ValueError("Expected exit1, exit2, and final logits")
    if len(alpha) != 3:
        raise ValueError("alpha must have three values")
    per_exit = {
        name: F.cross_entropy(logits[name], targets, weight=class_weights)
        for name in names
    }
    ce = sum(float(weight) * per_exit[name] for weight, name in zip(alpha, names))
    kd = torch.zeros((), dtype=ce.dtype, device=ce.device)
    if use_distillation and kd_gamma > 0:
        temperature = float(kd_temperature)
        teacher = torch.softmax(logits["final"].detach() / temperature, dim=1)
        for name in ("exit1", "exit2"):
            student = torch.log_softmax(logits[name] / temperature, dim=1)
            kd = kd + F.kl_div(student, teacher, reduction="batchmean") * temperature**2
        kd = kd / 2
    total = ce + float(kd_gamma) * kd
    return LossBreakdown(total=total, ce=ce, kd=kd, per_exit_ce=per_exit)

