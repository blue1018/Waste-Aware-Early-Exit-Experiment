from pathlib import Path

import pandas as pd
import pytest

from waste_early_exit.config import load_config
from waste_early_exit.experiments import (
    StageResult,
    aggregate_seed_comparisons,
    aggregate_seed_tables,
    build_seed_runners,
)


def test_full_mode_builds_one_isolated_runner_per_seed(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "full.yaml",
        project_root=Path(__file__).parents[1],
        overrides={
            "seeds": [7, 11, 19],
            "paths.artifact_root": str(tmp_path / "artifacts"),
        },
    )

    runners = build_seed_runners(config, project_root=Path(__file__).parents[1])

    assert tuple(runners) == (7, 11, 19)
    assert {runner.config.seed for runner in runners.values()} == {7, 11, 19}
    assert {runner.artifact_root.name for runner in runners.values()} == {
        "seed_7",
        "seed_11",
        "seed_19",
    }
    assert len({id(runner.progress) for runner in runners.values()}) == 1
    assert runners[7].progress.aggregate_log_path == tmp_path / "artifacts" / "aggregate" / "logs" / "full_run.log"


def test_smoke_mode_keeps_the_configured_artifact_root(tmp_path: Path) -> None:
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml",
        project_root=Path(__file__).parents[1],
        overrides={"paths.artifact_root": str(tmp_path / "artifacts")},
    )

    runners = build_seed_runners(config, project_root=Path(__file__).parents[1])

    assert tuple(runners) == (42,)
    assert runners[42].artifact_root == tmp_path / "artifacts"


def test_seed_comparisons_return_raw_values_and_confidence_intervals() -> None:
    stages = {
        1: StageResult(
            "locked_test",
            tables={
                "model_comparison": pd.DataFrame(
                    [{"method": "Proposed", "macro_f1": 0.80, "average_flops": 10.0}]
                )
            },
        ),
        2: StageResult(
            "locked_test",
            tables={
                "model_comparison": pd.DataFrame(
                    [{"method": "Proposed", "macro_f1": 0.90, "average_flops": 12.0}]
                )
            },
        ),
        3: StageResult(
            "locked_test",
            tables={
                "model_comparison": pd.DataFrame(
                    [{"method": "Proposed", "macro_f1": 0.85, "average_flops": 11.0}]
                )
            },
        ),
    }

    raw, summary = aggregate_seed_comparisons(stages)

    assert raw["seed"].tolist() == [1, 2, 3]
    assert summary.loc[0, "runs"] == 3
    assert summary.loc[0, "macro_f1_mean"] == pytest.approx(0.85)
    assert summary.loc[0, "macro_f1_std"] > 0
    assert summary.loc[0, "macro_f1_ci95"] > 0


def test_generic_seed_table_aggregates_latency_with_uncertainty() -> None:
    stages = {
        1: StageResult(
            "profiling",
            tables={"latency": pd.DataFrame([{"method": "Proposed", "runs": 25, "median_ms": 8.0}])},
        ),
        2: StageResult(
            "profiling",
            tables={"latency": pd.DataFrame([{"method": "Proposed", "runs": 25, "median_ms": 10.0}])},
        ),
        3: StageResult(
            "profiling",
            tables={"latency": pd.DataFrame([{"method": "Proposed", "runs": 25, "median_ms": 12.0}])},
        ),
    }

    raw, summary = aggregate_seed_tables(stages, "latency", ("method",))

    assert raw["seed"].tolist() == [1, 2, 3]
    assert summary.loc[0, "method"] == "Proposed"
    assert summary.loc[0, "runs"] == 3
    assert summary.loc[0, "benchmark_runs_mean"] == 25.0
    assert "runs_mean" not in summary.columns
    assert summary.loc[0, "median_ms_mean"] == 10.0
    assert summary.loc[0, "median_ms_std"] == 2.0
    assert summary.loc[0, "median_ms_ci95"] > 0
