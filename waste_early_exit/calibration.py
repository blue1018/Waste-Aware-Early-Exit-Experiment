from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .training import PredictionBundle


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    before_nll: float
    after_nll: float
    iterations: int


@dataclass(frozen=True)
class CalibratedBundle:
    logits: dict[str, torch.Tensor]
    probabilities: dict[str, torch.Tensor]
    temperatures: dict[str, float]
    labels: torch.Tensor
    sample_ids: list[str]
    fits: dict[str, TemperatureFit]


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iterations: int = 50,
) -> TemperatureFit:
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    targets = labels.detach().to(dtype=torch.long, device="cpu")
    before = float(F.cross_entropy(values, targets).item())
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=max_iterations,
        line_search_fn="strong_wolfe",
    )
    calls = 0

    def closure() -> torch.Tensor:
        nonlocal calls
        calls += 1
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(values / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).item())
    after = float(F.cross_entropy(values / temperature, targets).item())
    if after > before:
        temperature = 1.0
        after = before
    return TemperatureFit(temperature, before, after, calls)


def calibrate_bundle(bundle: PredictionBundle) -> CalibratedBundle:
    fits = {name: fit_temperature(logits, bundle.labels) for name, logits in bundle.logits.items()}
    temperatures = {name: fit.temperature for name, fit in fits.items()}
    probabilities = {
        name: torch.softmax(logits.detach().cpu() / temperatures[name], dim=1)
        for name, logits in bundle.logits.items()
    }
    return CalibratedBundle(
        logits={name: value.detach().cpu() for name, value in bundle.logits.items()},
        probabilities=probabilities,
        temperatures=temperatures,
        labels=bundle.labels.detach().cpu(),
        sample_ids=list(bundle.sample_ids),
        fits=fits,
    )

