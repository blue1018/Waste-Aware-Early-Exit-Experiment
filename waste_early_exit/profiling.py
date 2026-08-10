from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import torch
from torch import nn

from .reproducibility import synchronize_device


T = TypeVar("T")


@dataclass(frozen=True)
class EnergyRecord:
    status: str
    method: str
    energy_kwh: float | None
    energy_j: float | None
    duration_seconds: float
    detail: str


def weighted_dynamic_flops(exit_indices: torch.Tensor, exit_flops: Sequence[float]) -> float:
    indices = exit_indices.detach().cpu().numpy()
    values = np.asarray(exit_flops, dtype=np.float64)
    if indices.size == 0:
        raise ValueError("At least one exit index is required")
    if indices.min() < 0 or indices.max() >= len(values):
        raise ValueError("Exit index is outside the FLOPs table")
    return float(values[indices].mean())


class _ExitWrapper(nn.Module):
    def __init__(self, model: nn.Module, exit_name: str) -> None:
        super().__init__()
        self.model = model
        self.exit_name = exit_name

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "_to_layer2"):
            layer2 = self.model._to_layer2(inputs)
            if self.exit_name == "exit1":
                return self.model.exit1(layer2)
            layer3 = self.model.layer3(layer2)
            if self.exit_name == "exit2":
                return self.model.exit2(layer3)
            layer4 = self.model.layer4(layer3)
            return self.model.final(self.model.final_pool(layer4).flatten(1))
        output = self.model(inputs)
        return output["final"] if isinstance(output, dict) else output


def _hook_flops(model: nn.Module, inputs: torch.Tensor) -> float:
    total = 0.0
    handles: list[Any] = []

    def hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kernel = module.kernel_size[0] * module.kernel_size[1]
            operations = output.numel() * kernel * module.in_channels / module.groups
            total += float(operations)
        elif isinstance(module, nn.Linear):
            total += float(output.numel() * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(inputs)
    for handle in handles:
        handle.remove()
    return total


def estimate_exit_flops(
    model: nn.Module,
    input_shape: Sequence[int] = (1, 3, 224, 224),
    device: torch.device | None = None,
) -> dict[str, float]:
    target_device = device or torch.device("cpu")
    model = model.to(target_device).eval()
    inputs = torch.zeros(tuple(input_shape), device=target_device)
    exit_names = ("exit1", "exit2", "final") if hasattr(model, "exit1") else ("final",)
    results: dict[str, float] = {}
    for exit_name in exit_names:
        wrapper = _ExitWrapper(model, exit_name).to(target_device).eval()
        try:
            from fvcore.nn import FlopCountAnalysis

            results[exit_name] = float(FlopCountAnalysis(wrapper, inputs).unsupported_ops_warnings(False).total())
        except Exception:
            results[exit_name] = _hook_flops(wrapper, inputs)
    return results


def benchmark_latency(
    operation: Callable[[torch.Tensor], Any] | nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup_runs: int = 10,
    measured_runs: int = 100,
) -> dict[str, float | int | str]:
    values = inputs.to(device)
    if isinstance(operation, nn.Module):
        operation = operation.to(device).eval()
    with torch.inference_mode():
        for _ in range(warmup_runs):
            operation(values)
        synchronize_device(device)
        durations: list[float] = []
        for _ in range(measured_runs):
            started = time.perf_counter()
            operation(values)
            synchronize_device(device)
            durations.append((time.perf_counter() - started) * 1000)
    median_ms = float(np.median(durations))
    batch_size = int(values.shape[0]) if values.ndim > 0 else 1
    return {
        "device": str(device),
        "runs": int(measured_runs),
        "batch_size": batch_size,
        "median_ms": median_ms,
        "p95_ms": float(np.quantile(durations, 0.95)),
        "mean_ms": float(np.mean(durations)),
        "images_per_second": float(batch_size * 1000 / median_ms),
    }


def track_estimated_energy(
    operation: Callable[[], T],
    output_dir: str | Path,
    backend: str = "codecarbon",
) -> tuple[T, EnergyRecord]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if backend.lower() != "codecarbon":
        started = time.perf_counter()
        result = operation()
        duration = time.perf_counter() - started
        return result, EnergyRecord("unavailable", backend, None, None, duration, "No estimate backend was selected")
    tracker: Any | None = None
    started = time.perf_counter()
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(
            output_dir=str(destination),
            save_to_file=True,
            log_level="error",
            measure_power_secs=1,
            tracking_mode="process",
        )
        tracker.start()
    except Exception as error:
        result = operation()
        duration = time.perf_counter() - started
        return result, EnergyRecord("unavailable", "codecarbon", None, None, duration, f"{type(error).__name__}: {error}")
    try:
        result = operation()
    finally:
        emissions = tracker.stop()
    duration = time.perf_counter() - started
    data = getattr(tracker, "final_emissions_data", None)
    energy_kwh = getattr(data, "energy_consumed", None)
    if energy_kwh is None:
        return result, EnergyRecord(
            "unavailable",
            "codecarbon-estimate",
            None,
            None,
            duration,
            f"Emissions value was {emissions}",
        )
    value = float(energy_kwh)
    return result, EnergyRecord(
        "estimated",
        "codecarbon-estimate",
        value,
        value * 3_600_000,
        duration,
        "Software estimate; MPS power coverage may be incomplete",
    )


def parse_powermetrics(
    path: str | Path,
    sample_interval_seconds: float = 1.0,
) -> dict[str, float | int | str]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"Combined Power[^:]*:\s*([0-9.]+)\s*(mW|W)", re.IGNORECASE)
    samples: list[float] = []
    for value, unit in pattern.findall(text):
        watts = float(value) / 1000 if unit.lower() == "mw" else float(value)
        samples.append(watts)
    if not samples:
        raise ValueError("No combined power samples were found")
    duration = len(samples) * float(sample_interval_seconds)
    mean_power = float(np.mean(samples))
    return {
        "status": "measured",
        "method": "macOS powermetrics",
        "samples": len(samples),
        "sample_interval_seconds": float(sample_interval_seconds),
        "duration_seconds": duration,
        "mean_power_w": mean_power,
        "energy_j": mean_power * duration,
        "energy_kwh": mean_power * duration / 3_600_000,
    }


def compute_co2e(
    energy_kwh: float | None,
    carbon_intensity_g_per_kwh: float | None,
) -> float | None:
    if energy_kwh is None or carbon_intensity_g_per_kwh is None:
        return None
    return float(energy_kwh) * float(carbon_intensity_g_per_kwh)

