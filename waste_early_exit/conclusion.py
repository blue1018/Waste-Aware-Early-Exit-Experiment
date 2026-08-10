from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .config import ExperimentConfig


REQUIRED_METHODS = {
    "Proposed",
    "Global-Calibrated",
    "Global-Raw",
    "Fixed Exit 1",
    "Fixed Exit 2",
    "ResNet-18 Final-only",
    "Mobilenet V3 Small",
    "Efficientnet B0",
}


@dataclass(frozen=True)
class ConclusionAssessment:
    status: str
    checklist: pd.DataFrame
    blockers: tuple[str, ...]
    caveats: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "blockers": list(self.blockers),
            "caveats": list(self.caveats),
            "checklist": self.checklist.to_dict(orient="records"),
        }

    def save(self, output_root: str | Path) -> dict[str, Path]:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        checklist_path = root / "conclusion_checklist.csv"
        assessment_path = root / "conclusion_assessment.json"
        checklist_temporary = checklist_path.with_suffix(".csv.tmp")
        assessment_temporary = assessment_path.with_suffix(".json.tmp")
        self.checklist.to_csv(checklist_temporary, index=False)
        assessment_temporary.write_text(
            json.dumps(self.payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(checklist_temporary, checklist_path)
        os.replace(assessment_temporary, assessment_path)
        return {"checklist": checklist_path, "assessment": assessment_path}


def _all_artifacts(runners: Mapping[int, Any], relative_paths: tuple[str, ...]) -> tuple[bool, list[str]]:
    paths = [str(Path(runner.artifact_root) / relative) for runner in runners.values() for relative in relative_paths]
    return bool(paths) and all(Path(path).exists() for path in paths), paths


def _frozen_evidence(
    expected_seeds: set[int],
    locked_stages: Mapping[int, Any],
) -> tuple[bool, list[str], list[int]]:
    paths: list[str] = []
    invalid: list[int] = []
    for seed in sorted(expected_seeds):
        stage = locked_stages.get(seed)
        if stage is None:
            invalid.append(seed)
            continue
        path = stage.artifacts.get("frozen_manifest")
        if path is None or not Path(path).exists():
            invalid.append(seed)
            continue
        paths.append(str(path))
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(seed)
            continue
        if payload.get("test_evaluated") is not True:
            invalid.append(seed)
    return not invalid, paths, invalid


def assess_conclusion(
    config: ExperimentConfig,
    runners: Mapping[int, Any],
    locked_stages: Mapping[int, Any],
    seed_raw: pd.DataFrame,
    seed_summary: pd.DataFrame,
) -> ConclusionAssessment:
    expected_seeds = set(config.seeds if config.mode == "full" else (config.seed,))
    completed_seeds = set(int(seed) for seed in locked_stages)
    rows: list[dict[str, str]] = []

    def add(area: str, status: str, evidence: str, artifact: str = "") -> None:
        rows.append({"area": area, "status": status, "evidence": evidence, "artifact": artifact})

    missing_seeds = sorted(expected_seeds - completed_seeds)
    frozen_ok, frozen_paths, invalid_frozen = _frozen_evidence(expected_seeds, locked_stages)
    if missing_seeds or not frozen_ok:
        details = []
        if missing_seeds:
            details.append(f"missing seeds={missing_seeds}")
        if invalid_frozen:
            details.append(f"unfrozen or unevaluated seeds={invalid_frozen}")
        add("execution", "fail", "; ".join(details), ";".join(frozen_paths))
    else:
        add(
            "execution",
            "pass",
            f"all seeds completed and locked test followed freeze: {sorted(expected_seeds)}",
            ";".join(frozen_paths),
        )

    split_paths: list[str] = []
    data_failures: list[str] = []
    corrupt_count = 0
    conflict_count = 0
    for seed, runner in runners.items():
        path = Path(runner.artifact_root) / "splits" / "all_splits.csv"
        split_paths.append(str(path))
        if not path.exists():
            data_failures.append(f"seed {seed} missing split manifest")
            continue
        frame = pd.read_csv(path)
        required = {"relative_path", "duplicate_group", "split", "corrupt", "duplicate_class_conflict"}
        if not required.issubset(frame.columns):
            data_failures.append(f"seed {seed} split manifest lacks required columns")
            continue
        corrupt_count += int(frame["corrupt"].astype(bool).sum())
        conflict_count += int(frame["duplicate_class_conflict"].astype(bool).sum())
        valid = frame.loc[~frame["corrupt"].astype(bool)]
        if valid["relative_path"].duplicated().any():
            data_failures.append(f"seed {seed} repeats sample paths")
        if (valid.groupby("duplicate_group")["split"].nunique() > 1).any():
            data_failures.append(f"seed {seed} leaks duplicate groups across splits")
    if data_failures:
        add("data_integrity", "fail", "; ".join(data_failures), ";".join(split_paths))
    elif corrupt_count or conflict_count:
        add(
            "data_integrity",
            "caveat",
            f"no split leakage; corrupt rows={corrupt_count}; cross-class duplicate-conflict rows={conflict_count}",
            ";".join(split_paths),
        )
    else:
        add("data_integrity", "pass", "no corrupt rows, label conflicts, or split leakage", ";".join(split_paths))

    metric_columns = {
        "seed",
        "method",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "average_flops",
    }
    performance_ok = metric_columns.issubset(seed_raw.columns) and expected_seeds.issubset(
        set(seed_raw.get("seed", pd.Series(dtype=int)).astype(int))
    )
    add(
        "performance",
        "pass" if performance_ok else "fail",
        "aggregate predictive metrics include every seed" if performance_ok else "missing seed-level predictive metrics",
    )

    calibration_ok, calibration_paths = _all_artifacts(
        runners, ("results/calibration_self_distill.csv",)
    )
    add(
        "calibration",
        "pass" if calibration_ok else "fail",
        "per-exit temperature, NLL, and ECE artifact present" if calibration_ok else "missing calibration artifact",
        ";".join(calibration_paths),
    )

    routing_ok, routing_paths = _all_artifacts(
        runners,
        ("results/threshold_search_self_distill.csv", "results/class_difficulty_self_distill.csv"),
    )
    add(
        "routing",
        "pass" if routing_ok else "fail",
        "threshold search and class difficulty artifacts present" if routing_ok else "missing routing evidence",
        ";".join(routing_paths),
    )

    efficiency_ok, efficiency_paths = _all_artifacts(
        runners,
        (
            "results/flops_self_distill.csv",
            "results/latency_self_distill.csv",
            "results/energy_self_distill.csv",
        ),
    )
    latency_ratios: list[float] = []
    for runner in runners.values():
        latency_path = Path(runner.artifact_root) / "results" / "latency_self_distill.csv"
        if not latency_path.exists():
            continue
        latency = pd.read_csv(latency_path)
        if not {"method", "median_ms"}.issubset(latency.columns):
            continue
        indexed = latency.set_index("method")
        if {"Full ResNet-18", "Proposed dynamic"}.issubset(indexed.index):
            baseline_latency = float(indexed.loc["Full ResNet-18", "median_ms"])
            dynamic_latency = float(indexed.loc["Proposed dynamic", "median_ms"])
            if baseline_latency > 0:
                latency_ratios.append(dynamic_latency / baseline_latency)
    if latency_ratios:
        mean_ratio = sum(latency_ratios) / len(latency_ratios)
        if mean_ratio > 1:
            efficiency_evidence = (
                f"Proposed dynamic median latency is {mean_ratio:.2f}x slower than full ResNet-18; "
                "FLOPs and software-estimated energy cannot support a real efficiency claim"
            )
        else:
            efficiency_evidence = (
                f"Proposed dynamic median latency is {1 / mean_ratio:.2f}x faster than full ResNet-18; "
                "energy remains dependent on its measurement source"
            )
    else:
        efficiency_evidence = "FLOPs, latency, and energy artifacts exist but no comparable latency ratio is available"
    add(
        "efficiency",
        "caveat" if efficiency_ok else "fail",
        efficiency_evidence if efficiency_ok else "missing FLOPs, latency, or energy evidence",
        ";".join(efficiency_paths),
    )

    methods = set(seed_raw.get("method", pd.Series(dtype=str)).astype(str))
    missing_methods = sorted(REQUIRED_METHODS - methods)
    add(
        "comparability",
        "pass" if not missing_methods else "fail",
        "all required dynamic and static baselines present" if not missing_methods else f"missing methods={missing_methods}",
    )

    ablation_ok, ablation_paths = _all_artifacts(runners, ("results/ablations_self_distill.csv",))
    add(
        "ablations",
        "pass" if ablation_ok else "fail",
        "required ablation artifact present" if ablation_ok else "missing ablation evidence",
        ";".join(ablation_paths),
    )

    paired_ok, paired_paths = _all_artifacts(runners, ("results/paired_bootstrap_self_distill.csv",))
    uncertainty_columns = {"runs", "macro_f1_mean", "macro_f1_std", "macro_f1_ci95"}
    uncertainty_ok = uncertainty_columns.issubset(seed_summary.columns)
    correct_runs = bool(
        not seed_summary.empty
        and (seed_summary["runs"].astype(int) == len(expected_seeds)).all()
    ) if "runs" in seed_summary else False
    full_multiseed = config.mode == "full" and len(expected_seeds) >= 3
    statistics_ok = full_multiseed and paired_ok and uncertainty_ok and correct_runs
    if config.mode == "smoke":
        statistics_evidence = "single seed: paired bootstrap is diagnostic and cross-seed uncertainty is unavailable"
    elif statistics_ok:
        statistics_evidence = "cross-seed uncertainty and paired bootstrap evidence present"
    else:
        statistics_evidence = "three-seed uncertainty or paired bootstrap evidence is incomplete"
    add(
        "statistics",
        "pass" if statistics_ok else ("caveat" if config.mode == "smoke" else "fail"),
        statistics_evidence,
        ";".join(paired_paths),
    )

    add(
        "limitations",
        "caveat",
        "single dataset, class imbalance, duplicate-label conflicts, static thresholds, distribution shift, and Apple-Silicon energy limits must remain visible",
    )

    checklist = pd.DataFrame(rows)
    blockers = tuple(checklist.loc[checklist["status"] == "fail", "evidence"].astype(str))
    caveats = list(checklist.loc[checklist["status"] == "caveat", "evidence"].astype(str))
    if config.mode == "smoke":
        status = "pipeline validation only"
        caveats.insert(0, "Smoke mode validates the pipeline and cannot support a research conclusion")
    elif blockers:
        status = "incomplete full experiment"
    else:
        status = "full experiment ready with caveats"
    return ConclusionAssessment(status, checklist, blockers, tuple(caveats))
