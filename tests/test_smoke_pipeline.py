import json
from pathlib import Path

import pytest

from waste_early_exit.config import load_config
from waste_early_exit.experiments import ExperimentRunner


def read_log(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_config(tmp_path: Path, dataset_root: Path, image_size: int = 32):
    return load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml",
        project_root=Path(__file__).parents[1],
        overrides={
            "device": "cpu",
            "paths.dataset_root": str(dataset_root),
            "paths.artifact_root": str(tmp_path / "artifacts"),
            "data.image_size": image_size,
            "data.smoke_train_per_class": 1,
            "data.smoke_eval_per_class": 1,
            "model.pretrained": False,
            "training.epochs": 1,
            "training.static_epochs": 1,
            "training.batch_size": 4,
            "training.patience": 1,
            "routing.base_thresholds": [0.6, 0.99],
            "routing.lambdas": [0.0],
            "profiling.warmup_runs": 1,
            "profiling.latency_runs": 2,
            "profiling.energy_backend": "none",
        },
    )


def test_tiny_pipeline_emits_tables_figures_and_frozen_test_results(
    tmp_path: Path,
    small_image_dataset: dict[str, object],
) -> None:
    config = make_config(tmp_path, small_image_dataset["root"])
    runner = ExperimentRunner(config, project_root=Path(__file__).parents[1])

    setup = runner.setup()
    audit = runner.audit_data()
    split = runner.prepare_splits()
    eda = runner.run_eda()
    static = runner.train_static_baselines(["resnet18"])
    early = runner.train_early_exit_models(["ce"])
    calibration = runner.calibrate_variant("ce")
    search = runner.search_variant("ce")
    frozen = runner.freeze_variant("ce")
    locked = runner.evaluate_locked_test("ce", static_names=["resnet18"])
    ablations = runner.run_ablations("ce")
    profiling = runner.profile_variant("ce")
    runner.progress.close("complete")

    assert setup.validity == "pipeline validation only"
    assert audit.tables["overview"].loc[0, "classes"] == 12
    assert set(split.tables["split_counts"]["split"]) == {
        "train",
        "validation",
        "calibration",
        "threshold_validation",
        "test",
    }
    assert eda.figures and all(path.exists() for path in eda.figures.values())
    assert static.tables["training"].loc[0, "model"] == "resnet18"
    assert early.tables["training"].loc[0, "variant"] == "ce"
    assert set(calibration.tables["temperatures"]["exit"]) == {"exit1", "exit2", "final"}
    assert bool(search.tables["search"].iloc[0]["feasible"])
    assert frozen.artifacts["frozen_manifest"].exists()
    frozen_payload = json.loads(frozen.artifacts["frozen_manifest"].read_text(encoding="utf-8"))
    assert frozen_payload["checkpoint"] == "checkpoints/early_ce.pt"
    assert {"Proposed", "Global-Calibrated", "ResNet-18 Final-only"}.issubset(
        set(locked.tables["model_comparison"]["method"])
    )
    assert locked.tables["model_comparison"]["macro_f1"].between(0, 1).all()
    assert {
        "Proposed",
        "No temperature calibration",
        "No class awareness (lambda=0)",
        "No self-distillation",
        "Exit 2 only",
        "No difficulty shrinkage (rho=0)",
        "No recall guardrails",
    } == set(ablations.tables["ablations"]["method"])
    assert {"Global-Calibrated", "ResNet-18 Final-only"}.issubset(
        set(locked.tables["paired_bootstrap"]["baseline"])
    )
    assert locked.artifacts["paired_bootstrap"].exists()
    assert set(profiling.tables) >= {"flops", "latency", "energy"}
    assert profiling.tables["energy"].loc[0, "energy_kwh"] is None
    records = read_log(config.paths.artifact_root / "logs" / "run.log")
    kinds = {record["kind"] for record in records}
    assert {"stage_started", "stage_complete", "epoch_complete", "cache_miss"}.issubset(kinds)
    assert runner.progress.completed == 12


def test_audit_cache_is_reused_and_config_change_invalidates_it(
    tmp_path: Path,
    small_image_dataset: dict[str, object],
) -> None:
    first_config = make_config(tmp_path, small_image_dataset["root"], image_size=32)
    first = ExperimentRunner(first_config, project_root=Path(__file__).parents[1])
    first.setup()
    first_result = first.audit_data()
    second = ExperimentRunner(first_config, project_root=Path(__file__).parents[1])
    second.setup()
    second_result = second.audit_data()
    changed = ExperimentRunner(
        make_config(tmp_path, small_image_dataset["root"], image_size=40),
        project_root=Path(__file__).parents[1],
    )
    changed.setup()
    changed_result = changed.audit_data()

    assert not first_result.metadata["cache_hit"]
    assert second_result.metadata["cache_hit"]
    assert not changed_result.metadata["cache_hit"]


def test_training_cache_is_reused_and_config_change_invalidates_it(
    tmp_path: Path,
    small_image_dataset: dict[str, object],
) -> None:
    first_config = make_config(tmp_path, small_image_dataset["root"], image_size=32)
    first = ExperimentRunner(first_config, project_root=Path(__file__).parents[1])
    first.setup()
    first.audit_data()
    first.prepare_splits()
    first_result = first.train_static_baselines(["resnet18"])

    second = ExperimentRunner(first_config, project_root=Path(__file__).parents[1])
    second.setup()
    second.audit_data()
    second.prepare_splits()
    second_result = second.train_static_baselines(["resnet18"])

    changed_config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml",
        project_root=Path(__file__).parents[1],
        overrides={
            **first_config.normalized(),
            "paths.dataset_root": str(small_image_dataset["root"]),
            "paths.artifact_root": str(tmp_path / "artifacts"),
            "training.learning_rate": 0.002,
        },
    )
    changed = ExperimentRunner(changed_config, project_root=Path(__file__).parents[1])
    changed.setup()
    changed.audit_data()
    changed.prepare_splits()
    changed_result = changed.train_static_baselines(["resnet18"])

    assert not bool(first_result.tables["training"].loc[0, "cache_hit"])
    assert bool(second_result.tables["training"].loc[0, "cache_hit"])
    assert not bool(changed_result.tables["training"].loc[0, "cache_hit"])
    assert any(
        record["kind"] == "cache_hit"
        for record in read_log(first_config.paths.artifact_root / "logs" / "run.log")
    )


def test_locked_test_requires_frozen_manifest(
    tmp_path: Path,
    small_image_dataset: dict[str, object],
) -> None:
    runner = ExperimentRunner(
        make_config(tmp_path, small_image_dataset["root"]),
        project_root=Path(__file__).parents[1],
    )
    runner.setup()

    with pytest.raises(RuntimeError, match="frozen"):
        runner.evaluate_locked_test("ce", static_names=[])
