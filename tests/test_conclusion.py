import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from waste_early_exit.conclusion import assess_conclusion
from waste_early_exit.config import load_config
from waste_early_exit.experiments import StageResult


REQUIRED_METHODS = (
    "Proposed",
    "Global-Calibrated",
    "Global-Raw",
    "Fixed Exit 1",
    "Fixed Exit 2",
    "ResNet-18 Final-only",
    "Mobilenet V3 Small",
    "Efficientnet B0",
)


def make_config(tmp_path: Path, mode: str, seeds: list[int]):
    project_root = Path(__file__).parents[1]
    return load_config(
        project_root / "configs" / f"{mode}.yaml",
        project_root=project_root,
        overrides={"seeds": seeds, "paths.artifact_root": str(tmp_path / "artifacts")},
    )


def make_seed_evidence(root: Path, seed: int) -> tuple[SimpleNamespace, StageResult]:
    results = root / "results"
    splits = root / "splits"
    results.mkdir(parents=True)
    splits.mkdir(parents=True)
    frozen = results / "frozen_self_distill.json"
    frozen.write_text(json.dumps({"test_evaluated": True}), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "relative_path": "a.png",
                "duplicate_group": "g1",
                "split": "train",
                "corrupt": False,
                "duplicate_class_conflict": False,
            },
            {
                "relative_path": "b.png",
                "duplicate_group": "g2",
                "split": "test",
                "corrupt": False,
                "duplicate_class_conflict": False,
            },
        ]
    ).to_csv(splits / "all_splits.csv", index=False)
    for name in (
        "calibration_self_distill.csv",
        "threshold_search_self_distill.csv",
        "class_difficulty_self_distill.csv",
        "ablations_self_distill.csv",
        "flops_self_distill.csv",
        "latency_self_distill.csv",
        "energy_self_distill.csv",
        "paired_bootstrap_self_distill.csv",
    ):
        pd.DataFrame([{"seed": seed, "value": 1.0}]).to_csv(results / name, index=False)
    comparison = pd.DataFrame(
        [
            {
                "method": method,
                "accuracy": 0.8,
                "macro_precision": 0.8,
                "macro_recall": 0.8,
                "macro_f1": 0.8,
                "average_flops": 10.0,
            }
            for method in REQUIRED_METHODS
        ]
    )
    stage = StageResult(
        "locked_test",
        tables={"model_comparison": comparison},
        artifacts={"frozen_manifest": frozen},
    )
    return SimpleNamespace(artifact_root=root), stage


def aggregate_tables(seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(
        [
            {
                "seed": seed,
                "method": method,
                "accuracy": 0.8,
                "macro_precision": 0.8,
                "macro_recall": 0.8,
                "macro_f1": 0.8,
                "average_flops": 10.0,
            }
            for seed in seeds
            for method in REQUIRED_METHODS
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "method": method,
                "runs": len(seeds),
                "macro_f1_mean": 0.8,
                "macro_f1_std": 0.01,
                "macro_f1_ci95": 0.01,
                "average_flops_mean": 10.0,
            }
            for method in REQUIRED_METHODS
        ]
    )
    return raw, summary


def test_smoke_is_never_full_evidence(tmp_path: Path) -> None:
    config = make_config(tmp_path, "smoke", [42])
    runner, stage = make_seed_evidence(config.paths.artifact_root, 42)
    raw, summary = aggregate_tables([42])

    assessment = assess_conclusion(config, {42: runner}, {42: stage}, raw, summary)

    assert assessment.status == "pipeline validation only"
    assert any("smoke" in caveat.lower() for caveat in assessment.caveats)
    statistics = assessment.checklist.set_index("area").loc["statistics"]
    assert statistics["status"] == "caveat"
    assert "single seed" in statistics["evidence"].lower()


def test_full_missing_seed_is_incomplete(tmp_path: Path) -> None:
    config = make_config(tmp_path, "full", [1, 2, 3])
    runner, stage = make_seed_evidence(config.paths.artifact_root / "seed_1", 1)
    raw, summary = aggregate_tables([1])

    assessment = assess_conclusion(config, {1: runner}, {1: stage}, raw, summary)

    assert assessment.status == "incomplete full experiment"
    assert any("seed" in blocker.lower() for blocker in assessment.blockers)


def test_complete_three_seed_evidence_is_ready_with_caveats(tmp_path: Path) -> None:
    config = make_config(tmp_path, "full", [1, 2, 3])
    runners = {}
    stages = {}
    for seed in config.seeds:
        runner, stage = make_seed_evidence(config.paths.artifact_root / f"seed_{seed}", seed)
        runners[seed] = runner
        stages[seed] = stage
    raw, summary = aggregate_tables(list(config.seeds))

    assessment = assess_conclusion(config, runners, stages, raw, summary)

    assert assessment.status == "full experiment ready with caveats"
    assert not assessment.blockers
    assert set(assessment.checklist["area"]) == {
        "execution",
        "data_integrity",
        "performance",
        "calibration",
        "routing",
        "efficiency",
        "comparability",
        "ablations",
        "statistics",
        "limitations",
    }
    assert set(assessment.checklist["status"]) <= {"pass", "caveat", "fail"}


def test_assessment_save_exports_checklist_and_machine_readable_summary(tmp_path: Path) -> None:
    config = make_config(tmp_path, "smoke", [42])
    runner, stage = make_seed_evidence(config.paths.artifact_root, 42)
    raw, summary = aggregate_tables([42])
    assessment = assess_conclusion(config, {42: runner}, {42: stage}, raw, summary)

    paths = assessment.save(tmp_path / "aggregate" / "results")

    assert paths["checklist"].name == "conclusion_checklist.csv"
    assert paths["assessment"].name == "conclusion_assessment.json"
    saved = json.loads(paths["assessment"].read_text(encoding="utf-8"))
    assert saved["status"] == "pipeline validation only"
    assert len(pd.read_csv(paths["checklist"])) == 10


def test_efficiency_caveat_reports_when_dynamic_latency_is_slower(tmp_path: Path) -> None:
    config = make_config(tmp_path, "smoke", [42])
    runner, stage = make_seed_evidence(config.paths.artifact_root, 42)
    pd.DataFrame(
        [
            {"method": "Full ResNet-18", "median_ms": 5.0},
            {"method": "Proposed dynamic", "median_ms": 12.5},
        ]
    ).to_csv(config.paths.artifact_root / "results" / "latency_self_distill.csv", index=False)
    raw, summary = aggregate_tables([42])

    assessment = assess_conclusion(config, {42: runner}, {42: stage}, raw, summary)

    efficiency = assessment.checklist.set_index("area").loc["efficiency"]
    assert efficiency["status"] == "caveat"
    assert "2.50x slower" in efficiency["evidence"]
