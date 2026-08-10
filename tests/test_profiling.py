import math
from pathlib import Path

import torch

from waste_early_exit.profiling import (
    benchmark_latency,
    compute_co2e,
    parse_powermetrics,
    track_estimated_energy,
    weighted_dynamic_flops,
)


def test_weighted_dynamic_flops_matches_hand_calculation() -> None:
    exits = torch.tensor([0, 1, 2, 0])

    average = weighted_dynamic_flops(exits, [1.0, 2.0, 4.0])

    assert average == 2.0


def test_latency_summary_contains_median_p95_and_throughput() -> None:
    layer = torch.nn.Linear(4, 2).eval()
    inputs = torch.randn(1, 4)

    summary = benchmark_latency(layer, inputs, torch.device("cpu"), warmup_runs=1, measured_runs=5)

    assert summary["runs"] == 5
    assert summary["median_ms"] > 0
    assert summary["p95_ms"] >= summary["median_ms"]
    assert summary["images_per_second"] > 0


def test_unavailable_energy_backend_returns_missing_not_zero(tmp_path: Path) -> None:
    result, record = track_estimated_energy(lambda: "done", tmp_path, backend="none")

    assert result == "done"
    assert record.status == "unavailable"
    assert record.energy_kwh is None
    assert record.energy_j is None


def test_powermetrics_parser_integrates_combined_power_samples(tmp_path: Path) -> None:
    log_path = tmp_path / "power.txt"
    log_path.write_text(
        "Combined Power (CPU + GPU + ANE): 3000 mW\n"
        "Combined Power (CPU + GPU + ANE): 4000 mW\n",
        encoding="utf-8",
    )

    record = parse_powermetrics(log_path, sample_interval_seconds=1.0)

    assert record["samples"] == 2
    assert math.isclose(record["mean_power_w"], 3.5)
    assert math.isclose(record["energy_j"], 7.0)


def test_co2e_requires_both_energy_and_carbon_intensity() -> None:
    assert compute_co2e(0.001, 400.0) == 0.4
    assert compute_co2e(None, 400.0) is None
    assert compute_co2e(0.001, None) is None

