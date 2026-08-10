from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .calibration import CalibratedBundle, calibrate_bundle
from .config import ExperimentConfig
from .data import (
    assign_group_stratified_splits,
    build_dataloaders,
    build_manifest,
    class_names_from_frame,
    select_mode_subset,
)
from .eda import (
    plot_class_distribution,
    plot_image_sizes,
    plot_sample_grid,
    split_count_table,
    summarize_dataset,
)
from .losses import compute_class_weights
from .metrics import calibration_metrics, classification_metrics, paired_bootstrap, per_class_metrics
from .models import EarlyExitResNet18, build_static_model
from .profiling import (
    benchmark_latency,
    compute_co2e,
    estimate_exit_flops,
    track_estimated_energy,
    weighted_dynamic_flops,
)
from .progress import ProgressReporter, expected_stage_keys
from .reproducibility import environment_snapshot, resolve_device, seed_everything
from .routing import (
    RoutingResult,
    ThresholdSearchResult,
    make_class_thresholds,
    oracle_route,
    route_cached_logits,
    search_thresholds,
)
from .training import PredictionBundle, load_checkpoint, predict_logits, train_model
from .visualization import (
    plot_confusion_matrix,
    plot_exit_shares,
    plot_pareto,
    plot_per_class_metrics,
)


@dataclass(frozen=True)
class StageResult:
    name: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: dict[str, Path] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    validity: str = ""


_MethodResult = TypeVar("_MethodResult")


def tracked_stage(name: str) -> Callable[[Callable[..., _MethodResult]], Callable[..., _MethodResult]]:
    def decorator(method: Callable[..., _MethodResult]) -> Callable[..., _MethodResult]:
        @wraps(method)
        def wrapper(self: "ExperimentRunner", *args: Any, **kwargs: Any) -> _MethodResult:
            with self.progress.stage(self.config.seed, name):
                return method(self, *args, **kwargs)

        return wrapper

    return decorator


def build_seed_runners(
    config: ExperimentConfig,
    project_root: str | Path | None = None,
    progress: ProgressReporter | None = None,
    progress_display: bool = True,
) -> dict[int, "ExperimentRunner"]:
    seeds = config.seeds if config.mode == "full" else (config.seed,)
    artifact_roots = {
        seed: config.paths.artifact_root / f"seed_{seed}" if len(seeds) > 1 else config.paths.artifact_root
        for seed in seeds
    }
    shared_progress = progress or ProgressReporter(
        expected_stage_keys(config.mode, seeds),
        config.paths.artifact_root / "aggregate" / "logs" / f"{config.mode}_run.log",
        {seed: root / "logs" / "run.log" for seed, root in artifact_roots.items()},
        display=progress_display,
    )
    runners: dict[int, ExperimentRunner] = {}
    for seed in seeds:
        artifact_root = artifact_roots[seed]
        seed_paths = replace(config.paths, artifact_root=artifact_root)
        seed_config = replace(config, seed=seed, paths=seed_paths)
        runners[seed] = ExperimentRunner(
            seed_config,
            project_root=project_root,
            progress=shared_progress,
        )
    return runners


def aggregate_seed_comparisons(
    stages: dict[int, StageResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed, stage in stages.items():
        frame = stage.tables["model_comparison"].copy()
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        raise ValueError("At least one locked-test stage is required")
    raw = pd.concat(frames, ignore_index=True)
    metric_columns = [
        column
        for column in raw.select_dtypes(include=[np.number]).columns
        if column != "seed"
    ]
    rows: list[dict[str, Any]] = []
    for method, group in raw.groupby("method", sort=False):
        row: dict[str, Any] = {"method": method, "runs": len(group)}
        for column in metric_columns:
            values = group[column].astype(float)
            standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = standard_deviation
            row[f"{column}_ci95"] = 1.96 * standard_deviation / np.sqrt(len(values))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if "macro_f1_mean" in summary.columns:
        summary = summary.sort_values("macro_f1_mean", ascending=False).reset_index(drop=True)
    return raw, summary


def aggregate_seed_tables(
    stages: Mapping[int, StageResult],
    table_name: str,
    group_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for seed, stage in stages.items():
        if table_name not in stage.tables:
            raise KeyError(f"Stage {stage.name!r} has no table {table_name!r}")
        frame = stage.tables[table_name].copy()
        frame["seed"] = int(seed)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one stage is required")
    raw = pd.concat(frames, ignore_index=True)
    if "runs" in raw.columns and "runs" not in group_columns:
        raw = raw.rename(columns={"runs": "benchmark_runs"})
    missing = set(group_columns) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing grouping columns: {sorted(missing)}")
    metric_columns = [
        column
        for column in raw.select_dtypes(include=[np.number]).columns
        if column != "seed" and column not in group_columns
    ]
    if not metric_columns:
        raise ValueError(f"Table {table_name!r} has no numeric metric columns")
    grouper: str | list[str]
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    rows: list[dict[str, Any]] = []
    for group_key, group in raw.groupby(grouper, sort=False, dropna=False):
        values = (group_key,) if len(group_columns) == 1 else tuple(group_key)
        row: dict[str, Any] = dict(zip(group_columns, values))
        row["runs"] = int(group["seed"].nunique())
        for column in metric_columns:
            series = group[column].astype(float)
            standard_deviation = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            row[f"{column}_mean"] = float(series.mean())
            row[f"{column}_std"] = standard_deviation
            row[f"{column}_ci95"] = 1.96 * standard_deviation / np.sqrt(len(series))
        rows.append(row)
    return raw, pd.DataFrame(rows)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_path(root: Path, model_key: str, split: str) -> Path:
    safe = model_key.replace(":", "_").replace("/", "_")
    return root / f"{safe}_{split}.pt"


def _save_bundle(path: Path, bundle: PredictionBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "logits": bundle.logits,
            "labels": bundle.labels,
            "sample_ids": bundle.sample_ids,
        },
        temporary,
    )
    os.replace(temporary, path)


def _load_bundle(path: Path) -> PredictionBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return PredictionBundle(payload["logits"], payload["labels"], list(payload["sample_ids"]))


def _fixed_route(logits: torch.Tensor, exit_index: int) -> RoutingResult:
    values = logits.detach().cpu()
    probabilities = torch.softmax(values, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    exits = torch.full((len(values),), exit_index, dtype=torch.long)
    return RoutingResult(predictions, exits, confidence, values)


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        project_root: str | Path | None = None,
        progress: ProgressReporter | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.artifact_root = config.paths.artifact_root
        self.progress = progress or ProgressReporter(
            expected_stage_keys(config.mode, (config.seed,)),
            self.artifact_root / "aggregate" / "logs" / f"{config.mode}_run.log",
            {config.seed: self.artifact_root / "logs" / "run.log"},
            display=False,
        )
        self.device = torch.device("cpu")
        self.validity = "pipeline validation only" if config.mode == "smoke" else "full experiment"
        self.manifest: pd.DataFrame | None = None
        self.split_df: pd.DataFrame | None = None
        self.mode_df: pd.DataFrame | None = None
        self.loaders: dict[str, Any] = {}
        self.class_names: list[str] = []
        self.models: dict[str, torch.nn.Module] = {}
        self.predictions: dict[str, PredictionBundle] = {}
        self.calibrations: dict[str, CalibratedBundle] = {}
        self.searches: dict[str, ThresholdSearchResult] = {}
        self.flops: dict[str, dict[str, float]] = {}
        self._audit_signature = ""

    def _directories(self) -> dict[str, Path]:
        return {
            name: self.artifact_root / name
            for name in (
                "manifests",
                "splits",
                "checkpoints",
                "cached_logits",
                "results",
                "figures",
                "logs",
            )
        }

    @tracked_stage("setup")
    def setup(self) -> StageResult:
        for path in self._directories().values():
            path.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(self.config.device)
        seed_everything(self.config.seed)
        snapshot = environment_snapshot(self.device)
        snapshot.update(
            {
                "mode": self.config.mode,
                "config_fingerprint": self.config.fingerprint(),
                "dataset_root": str(self.config.paths.dataset_root),
            }
        )
        path = self._directories()["logs"] / "environment.json"
        _write_json_atomic(path, snapshot)
        table = pd.DataFrame([snapshot])
        return StageResult(
            "setup",
            tables={"environment": table},
            artifacts={"environment": path},
            metadata=snapshot,
            message=f"Running {self.config.mode} on {self.device}",
            validity=self.validity,
        )

    def _dataset_inventory_hash(self) -> str:
        root = self.config.paths.dataset_root
        digest = hashlib.sha256(self.config.fingerprint().encode("utf-8"))
        for path in sorted(value for value in root.rglob("*") if value.is_file()):
            stat = path.stat()
            record = f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n"
            digest.update(record.encode("utf-8"))
        return digest.hexdigest()

    @tracked_stage("audit")
    def audit_data(self, force: bool = False) -> StageResult:
        directories = self._directories()
        manifest_path = directories["manifests"] / "manifest.csv"
        metadata_path = directories["manifests"] / "manifest.meta.json"
        signature = self._dataset_inventory_hash()
        cache_hit = False
        if not force and manifest_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                self.manifest = pd.read_csv(manifest_path)
                cache_hit = True
        if self.manifest is None:
            self.manifest = build_manifest(
                self.config.paths.dataset_root,
                phash_distance=self.config.data.phash_distance,
            )
            _write_csv_atomic(manifest_path, self.manifest)
            _write_json_atomic(metadata_path, {"signature": signature})
        self._audit_signature = signature
        corrupt_share = float(self.manifest["corrupt"].mean()) if len(self.manifest) else 0.0
        if corrupt_share > self.config.data.corruption_tolerance:
            raise RuntimeError(
                f"Corrupt image share {corrupt_share:.2%} exceeds the configured tolerance"
            )
        summary = summarize_dataset(self.manifest)
        return StageResult(
            "audit",
            tables=summary,
            artifacts={"manifest": manifest_path, "metadata": metadata_path},
            metadata={"cache_hit": cache_hit, "signature": signature},
            message=f"Found {len(self.manifest):,} image files",
            validity=self.validity,
        )

    @tracked_stage("splits")
    def prepare_splits(self, force: bool = False) -> StageResult:
        if self.manifest is None:
            self.audit_data()
        directories = self._directories()
        split_path = directories["splits"] / "all_splits.csv"
        mode_path = directories["splits"] / f"{self.config.mode}_subset.csv"
        metadata_path = directories["splits"] / f"{self.config.mode}_splits.meta.json"
        signature = hashlib.sha256(
            f"{self._audit_signature}|{self.config.fingerprint()}|splits".encode("utf-8")
        ).hexdigest()
        cache_hit = False
        if not force and split_path.exists() and mode_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                self.split_df = pd.read_csv(split_path)
                self.mode_df = pd.read_csv(mode_path)
                cache_hit = True
        if self.split_df is None or self.mode_df is None:
            self.split_df = assign_group_stratified_splits(self.manifest, self.config.seed)
            self.mode_df = select_mode_subset(self.split_df, self.config)
            _write_csv_atomic(split_path, self.split_df)
            _write_csv_atomic(mode_path, self.mode_df)
            _write_json_atomic(metadata_path, {"signature": signature})
        self.class_names = class_names_from_frame(self.mode_df)
        self.loaders = build_dataloaders(self.mode_df, self.config)
        counts = self.mode_df.groupby("split").size().rename("images").reset_index()
        class_counts = split_count_table(self.mode_df.assign(corrupt=False))
        return StageResult(
            "splits",
            tables={"split_counts": counts, "class_split_counts": class_counts},
            artifacts={"all_splits": split_path, "mode_subset": mode_path},
            metadata={"cache_hit": cache_hit, "signature": signature},
            message="Frozen leakage-safe data splits",
            validity=self.validity,
        )

    @tracked_stage("eda")
    def run_eda(self) -> StageResult:
        if self.manifest is None:
            self.audit_data()
        if self.mode_df is None:
            self.prepare_splits()
        figure_root = self._directories()["figures"] / "eda"
        paths = {
            "class_distribution": figure_root / "class_distribution.png",
            "image_sizes": figure_root / "image_sizes.png",
            "sample_grid": figure_root / "sample_grid.png",
        }
        figures = [
            plot_class_distribution(self.manifest, paths["class_distribution"]),
            plot_image_sizes(self.manifest, paths["image_sizes"]),
            plot_sample_grid(self.manifest, paths["sample_grid"], self.config.seed),
        ]
        for figure in figures:
            plt.close(figure)
        tables = summarize_dataset(self.manifest)
        tables["split_counts"] = self.mode_df.groupby("split").size().rename("images").reset_index()
        return StageResult(
            "eda",
            tables=tables,
            figures=paths,
            message="Dataset audit and EDA are ready",
            validity=self.validity,
        )

    def _class_weights(self) -> torch.Tensor:
        if self.mode_df is None:
            self.prepare_splits()
        train = self.mode_df.loc[self.mode_df["split"] == "train"]
        counts = (
            train.groupby("class_index").size().reindex(range(len(self.class_names)), fill_value=0).to_numpy()
        )
        return compute_class_weights(
            torch.tensor(counts),
            self.config.model.class_weight_min,
            self.config.model.class_weight_max,
        )

    def _predict_and_cache(self, key: str, model: torch.nn.Module) -> None:
        for split in ("validation", "calibration", "threshold_validation", "test"):
            label = f"{key}:{split}"
            self.progress.event(
                "prediction_started",
                self.config.seed,
                stage="prediction_cache",
                model=key,
                split=split,
            )
            bundle = predict_logits(
                model,
                self.loaders[split],
                self.device,
                progress=self.progress,
                progress_seed=self.config.seed,
                progress_label=label,
            )
            self.predictions[f"{key}:{split}"] = bundle
            _save_bundle(_bundle_path(self._directories()["cached_logits"], key, split), bundle)
            self.progress.event(
                "prediction_complete",
                self.config.seed,
                stage="prediction_cache",
                model=key,
                split=split,
                samples=len(bundle.labels),
            )

    def _training_signature(self, model_key: str) -> str:
        payload = f"{self._audit_signature}|{self.config.fingerprint()}|{model_key}|training"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @tracked_stage("static_training")
    def train_static_baselines(self, names: Sequence[str] | None = None) -> StageResult:
        if not self.loaders:
            self.prepare_splits()
        selected_names = list(names or ("mobilenet_v3_small", "efficientnet_b0", "resnet18"))
        rows: list[dict[str, Any]] = []
        histories: list[pd.DataFrame] = []
        class_weights = self._class_weights()
        for name in selected_names:
            self.progress.update(model=name, device=str(self.device))
            key = f"static:{name}"
            checkpoint = self._directories()["checkpoints"] / f"{name}.pt"
            history_path = self._directories()["results"] / f"training_{name}.csv"
            metadata_path = self._directories()["results"] / f"training_{name}.meta.json"
            signature = self._training_signature(key)
            model = build_static_model(name, len(self.class_names), self.config.model.pretrained)
            cache_hit = False
            if checkpoint.exists() and history_path.exists() and metadata_path.exists():
                cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                cache_hit = cache_metadata.get("signature") == signature
            self.progress.event(
                "cache_hit" if cache_hit else "cache_miss",
                self.config.seed,
                stage="static_training",
                model=name,
                checkpoint=str(checkpoint),
            )
            if cache_hit:
                metadata = load_checkpoint(checkpoint, model, map_location=self.device)
                history = pd.read_csv(history_path)
                best_metric = float(metadata["best_metric"])
                best_epoch = int(metadata["epoch"])
                batch_size = int(metadata["effective_batch_size"])
                eta = float("nan")
            else:
                result = train_model(
                    model,
                    self.loaders["train"],
                    self.loaders["validation"],
                    self.config,
                    self.device,
                    checkpoint,
                    class_weights=class_weights,
                    use_distillation=False,
                    epochs=self.config.training.static_epochs,
                    progress=self.progress,
                    progress_seed=self.config.seed,
                    progress_label=name,
                )
                history = result.history
                _write_csv_atomic(history_path, history)
                _write_json_atomic(metadata_path, {"signature": signature})
                best_metric = result.best_metric
                best_epoch = result.best_epoch
                batch_size = result.effective_batch_size
                eta = result.estimated_total_seconds
            self.models[key] = model.to(self.device)
            self._predict_and_cache(key, self.models[key])
            history_with_model = history.copy()
            history_with_model["model"] = name
            histories.append(history_with_model)
            rows.append(
                {
                    "model": name,
                    "best_validation_macro_f1": best_metric,
                    "best_epoch": best_epoch,
                    "batch_size": batch_size,
                    "estimated_total_seconds": eta,
                    "cache_hit": cache_hit,
                }
            )
        return StageResult(
            "static_training",
            tables={"training": pd.DataFrame(rows), "history": pd.concat(histories, ignore_index=True)},
            artifacts={"checkpoint_root": self._directories()["checkpoints"]},
            message="Static baselines are trained",
            validity=self.validity,
        )

    @tracked_stage("early_training")
    def train_early_exit_models(self, variants: Sequence[str] | None = None) -> StageResult:
        if not self.loaders:
            self.prepare_splits()
        selected_variants = list(variants or ("ce", "self_distill"))
        unknown = set(selected_variants) - {"ce", "self_distill"}
        if unknown:
            raise ValueError(f"Unknown early-exit variants: {sorted(unknown)}")
        rows: list[dict[str, Any]] = []
        histories: list[pd.DataFrame] = []
        class_weights = self._class_weights()
        for variant in selected_variants:
            self.progress.update(variant=variant, model=f"early_{variant}", device=str(self.device))
            key = f"early:{variant}"
            checkpoint = self._directories()["checkpoints"] / f"early_{variant}.pt"
            history_path = self._directories()["results"] / f"training_early_{variant}.csv"
            metadata_path = self._directories()["results"] / f"training_early_{variant}.meta.json"
            signature = self._training_signature(key)
            model = EarlyExitResNet18(
                len(self.class_names),
                pretrained=self.config.model.pretrained,
                dropout=self.config.model.dropout,
            )
            cache_hit = False
            if checkpoint.exists() and history_path.exists() and metadata_path.exists():
                cache_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                cache_hit = cache_metadata.get("signature") == signature
            self.progress.event(
                "cache_hit" if cache_hit else "cache_miss",
                self.config.seed,
                stage="early_training",
                model=f"early_{variant}",
                variant=variant,
                checkpoint=str(checkpoint),
            )
            if cache_hit:
                metadata = load_checkpoint(checkpoint, model, map_location=self.device)
                history = pd.read_csv(history_path)
                best_metric = float(metadata["best_metric"])
                best_epoch = int(metadata["epoch"])
                batch_size = int(metadata["effective_batch_size"])
                eta = float("nan")
            else:
                result = train_model(
                    model,
                    self.loaders["train"],
                    self.loaders["validation"],
                    self.config,
                    self.device,
                    checkpoint,
                    class_weights=class_weights,
                    use_distillation=variant == "self_distill",
                    progress=self.progress,
                    progress_seed=self.config.seed,
                    progress_label=f"early_{variant}",
                )
                history = result.history
                _write_csv_atomic(history_path, history)
                _write_json_atomic(metadata_path, {"signature": signature})
                best_metric = result.best_metric
                best_epoch = result.best_epoch
                batch_size = result.effective_batch_size
                eta = result.estimated_total_seconds
            self.models[key] = model.to(self.device)
            self._predict_and_cache(key, self.models[key])
            history_with_variant = history.copy()
            history_with_variant["variant"] = variant
            histories.append(history_with_variant)
            rows.append(
                {
                    "variant": variant,
                    "best_validation_macro_f1": best_metric,
                    "best_epoch": best_epoch,
                    "batch_size": batch_size,
                    "estimated_total_seconds": eta,
                    "cache_hit": cache_hit,
                }
            )
        return StageResult(
            "early_training",
            tables={"training": pd.DataFrame(rows), "history": pd.concat(histories, ignore_index=True)},
            artifacts={"checkpoint_root": self._directories()["checkpoints"]},
            message="Early-exit models are trained",
            validity=self.validity,
        )

    def _ensure_variant(self, variant: str) -> str:
        key = f"early:{variant}"
        if key not in self.models:
            self.train_early_exit_models([variant])
        return key

    @tracked_stage("calibration")
    def calibrate_variant(self, variant: str = "self_distill") -> StageResult:
        key = self._ensure_variant(variant)
        bundle = self.predictions[f"{key}:calibration"]
        calibrated = calibrate_bundle(bundle)
        self.calibrations[variant] = calibrated
        rows = []
        for exit_name, fit in calibrated.fits.items():
            rows.append(
                {
                    "exit": exit_name,
                    "temperature": fit.temperature,
                    "nll_before": fit.before_nll,
                    "nll_after": fit.after_nll,
                    "ece_after": calibration_metrics(
                        bundle.logits[exit_name] / fit.temperature,
                        bundle.labels,
                    )["ece"],
                }
            )
        table = pd.DataFrame(rows)
        output_path = self._directories()["results"] / f"calibration_{variant}.csv"
        _write_csv_atomic(output_path, table)
        return StageResult(
            "calibration",
            tables={"temperatures": table},
            artifacts={"calibration": output_path},
            message="Each exit now has its own temperature",
            validity=self.validity,
        )

    @tracked_stage("threshold_search")
    def search_variant(self, variant: str = "self_distill") -> StageResult:
        key = self._ensure_variant(variant)
        if variant not in self.calibrations:
            self.calibrate_variant(variant)
        calibration = self.calibrations[variant]
        bundle = self.predictions[f"{key}:threshold_validation"]
        final_predictions = bundle.logits["final"].argmax(dim=1)
        final_overall = classification_metrics(bundle.labels.numpy(), final_predictions.numpy())
        final_classes = per_class_metrics(bundle.labels.numpy(), final_predictions.numpy(), self.class_names)
        macro_floor = max(0.0, final_overall["macro_f1"] - self.config.routing.macro_f1_tolerance)
        recall_floors = []
        for _, row in final_classes.iterrows():
            tolerance = (
                self.config.routing.battery_recall_tolerance
                if row["class_name"] == "battery"
                else self.config.routing.recall_tolerance
            )
            recall_floors.append(max(0.0, float(row["recall"]) - tolerance))
        if key not in self.flops:
            self.flops[key] = estimate_exit_flops(
                self.models[key],
                (1, 3, self.config.data.image_size, self.config.data.image_size),
                self.device,
            )
        exit_flops = [self.flops[key][name] for name in ("exit1", "exit2", "final")]
        search = search_thresholds(
            logits=bundle.logits,
            labels=bundle.labels,
            temperatures=calibration.temperatures,
            class_names=self.class_names,
            base_threshold_grid=self.config.routing.base_thresholds,
            lambda_grid=self.config.routing.lambdas,
            rho=self.config.routing.rho,
            exit_flops=exit_flops,
            macro_f1_floor=macro_floor,
            recall_floors=recall_floors,
            split_name="threshold_validation",
            minimum=self.config.routing.min_threshold,
            maximum=self.config.routing.max_threshold,
        )
        self.searches[variant] = search
        search_path = self._directories()["results"] / f"threshold_search_{variant}.csv"
        difficulty_path = self._directories()["results"] / f"class_difficulty_{variant}.csv"
        _write_csv_atomic(search_path, search.table)
        _write_csv_atomic(difficulty_path, search.difficulty)
        threshold_rows = []
        for exit_name, values in search.thresholds.items():
            for class_index, value in enumerate(values.tolist()):
                threshold_rows.append(
                    {
                        "exit": exit_name,
                        "class_index": class_index,
                        "class_name": self.class_names[class_index],
                        "threshold": value,
                    }
                )
        return StageResult(
            "threshold_search",
            tables={
                "search": search.table,
                "difficulty": search.difficulty,
                "selected_thresholds": pd.DataFrame(threshold_rows),
            },
            artifacts={"search": search_path, "difficulty": difficulty_path},
            metadata={"macro_f1_floor": macro_floor, "recall_floors": recall_floors},
            message="Selected the lowest-FLOPs feasible class-aware thresholds",
            validity=self.validity,
        )

    @tracked_stage("freeze")
    def freeze_variant(self, variant: str = "self_distill") -> StageResult:
        key = self._ensure_variant(variant)
        if variant not in self.searches:
            self.search_variant(variant)
        checkpoint = self._directories()["checkpoints"] / f"early_{variant}.pt"
        search = self.searches[variant]
        calibration = self.calibrations[variant]
        payload = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "variant": variant,
            "config_fingerprint": self.config.fingerprint(),
            "data_fingerprint": self._audit_signature,
            "checkpoint": checkpoint.relative_to(self.artifact_root).as_posix(),
            "checkpoint_sha256": _sha256_file(checkpoint),
            "temperatures": calibration.temperatures,
            "thresholds": {name: values.tolist() for name, values in search.thresholds.items()},
            "best_search_row": search.best_row,
            "test_evaluated": False,
        }
        path = self._directories()["results"] / f"frozen_{variant}.json"
        _write_json_atomic(path, payload)
        return StageResult(
            "freeze",
            tables={"frozen_settings": pd.DataFrame([search.best_row])},
            artifacts={"frozen_manifest": path},
            metadata=payload,
            message="Model, calibration, and routing settings are frozen",
            validity=self.validity,
        )

    @tracked_stage("locked_test")
    def evaluate_locked_test(
        self,
        variant: str = "self_distill",
        static_names: Sequence[str] | None = None,
    ) -> StageResult:
        frozen_path = self._directories()["results"] / f"frozen_{variant}.json"
        if not frozen_path.exists():
            raise RuntimeError("A frozen manifest is required before locked-test evaluation")
        key = self._ensure_variant(variant)
        if variant not in self.searches or variant not in self.calibrations:
            raise RuntimeError("The current runner must freeze calibration and threshold state before testing")
        bundle = self.predictions[f"{key}:test"]
        search = self.searches[variant]
        calibrated_temperatures = self.calibrations[variant].temperatures
        best = search.best_row
        uniform = {
            "exit1": torch.full((len(self.class_names),), float(best["tau_exit1"])),
            "exit2": torch.full((len(self.class_names),), float(best["tau_exit2"])),
        }
        methods: dict[str, RoutingResult] = {
            "Fixed Exit 1": _fixed_route(bundle.logits["exit1"], 0),
            "Fixed Exit 2": _fixed_route(bundle.logits["exit2"], 1),
            "Global-Raw": route_cached_logits(
                bundle.logits,
                {"exit1": 1.0, "exit2": 1.0, "final": 1.0},
                uniform,
            ),
            "Global-Calibrated": route_cached_logits(bundle.logits, calibrated_temperatures, uniform),
            "Proposed": route_cached_logits(bundle.logits, calibrated_temperatures, search.thresholds),
            "Oracle": oracle_route(bundle.logits, bundle.labels),
        }
        method_predictions: dict[str, np.ndarray] = {
            method: routed.predictions.numpy() for method, routed in methods.items()
        }
        exit_flops = [self.flops[key][name] for name in ("exit1", "exit2", "final")]
        rows: list[dict[str, Any]] = []
        class_tables: list[pd.DataFrame] = []
        exit_rows: list[dict[str, Any]] = []
        for method, routed in methods.items():
            overall = classification_metrics(bundle.labels.numpy(), routed.predictions.numpy())
            row = {"method": method, **overall}
            row["average_flops"] = weighted_dynamic_flops(routed.exit_indices, exit_flops)
            rows.append(row)
            classes = per_class_metrics(bundle.labels.numpy(), routed.predictions.numpy(), self.class_names)
            classes["method"] = method
            class_tables.append(classes)
            counts = torch.bincount(routed.exit_indices, minlength=3).numpy() / len(routed.exit_indices)
            exit_rows.append(
                {"method": method, "exit1": counts[0], "exit2": counts[1], "final": counts[2]}
            )

        requested_static = list(static_names or ("mobilenet_v3_small", "efficientnet_b0", "resnet18"))
        for name in requested_static:
            static_key = f"static:{name}"
            if static_key not in self.models:
                self.train_static_baselines([name])
            static_bundle = self.predictions[f"{static_key}:test"]
            if static_bundle.sample_ids != bundle.sample_ids:
                raise AssertionError(f"Static test sample order differs for paired comparison: {name}")
            predictions = static_bundle.logits["final"].argmax(dim=1)
            overall = classification_metrics(static_bundle.labels.numpy(), predictions.numpy())
            display_name = "ResNet-18 Final-only" if name == "resnet18" else name.replace("_", " ").title()
            method_predictions[display_name] = predictions.numpy()
            static_flops = estimate_exit_flops(
                self.models[static_key],
                (1, 3, self.config.data.image_size, self.config.data.image_size),
                self.device,
            )["final"]
            rows.append({"method": display_name, **overall, "average_flops": static_flops})
            classes = per_class_metrics(static_bundle.labels.numpy(), predictions.numpy(), self.class_names)
            classes["method"] = display_name
            class_tables.append(classes)
        comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
        class_comparison = pd.concat(class_tables, ignore_index=True)
        exits = pd.DataFrame(exit_rows)
        bootstrap_rows: list[dict[str, Any]] = []
        proposed_predictions = method_predictions["Proposed"]
        for baseline in (
            "Global-Calibrated",
            "ResNet-18 Final-only",
            "Mobilenet V3 Small",
            "Efficientnet B0",
        ):
            if baseline not in method_predictions:
                continue
            comparison_result = paired_bootstrap(
                bundle.labels.numpy(),
                proposed_predictions,
                method_predictions[baseline],
                n_resamples=500 if self.config.mode == "smoke" else 2000,
                seed=self.config.seed,
            )
            bootstrap_rows.append(
                {
                    "seed": self.config.seed,
                    "method": "Proposed",
                    "baseline": baseline,
                    **comparison_result,
                }
            )
        bootstrap_table = pd.DataFrame(bootstrap_rows)
        results_root = self._directories()["results"]
        comparison_path = results_root / f"locked_test_comparison_{variant}.csv"
        class_path = results_root / f"locked_test_per_class_{variant}.csv"
        bootstrap_path = results_root / f"paired_bootstrap_{variant}.csv"
        _write_csv_atomic(comparison_path, comparison)
        _write_csv_atomic(class_path, class_comparison)
        _write_csv_atomic(bootstrap_path, bootstrap_table)
        figure_root = self._directories()["figures"] / "results"
        figure_paths = {
            "proposed_confusion": figure_root / "proposed_confusion.png",
            "exit_shares": figure_root / "exit_shares.png",
            "flops_pareto": figure_root / "flops_pareto.png",
            "per_class_f1": figure_root / "per_class_f1.png",
        }
        figures = [
            plot_confusion_matrix(
                bundle.labels.numpy(),
                methods["Proposed"].predictions.numpy(),
                self.class_names,
                figure_paths["proposed_confusion"],
            ),
            plot_exit_shares(exits, figure_paths["exit_shares"]),
            plot_pareto(comparison, "average_flops", "macro_f1", figure_paths["flops_pareto"]),
            plot_per_class_metrics(class_comparison, "f1", figure_paths["per_class_f1"]),
        ]
        for figure in figures:
            plt.close(figure)
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen["test_evaluated"] = True
        frozen["test_evaluated_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(frozen_path, frozen)
        return StageResult(
            "locked_test",
            tables={
                "model_comparison": comparison,
                "per_class": class_comparison,
                "exit_shares": exits,
                "paired_bootstrap": bootstrap_table,
            },
            figures=figure_paths,
            artifacts={
                "comparison": comparison_path,
                "per_class": class_path,
                "paired_bootstrap": bootstrap_path,
                "frozen_manifest": frozen_path,
            },
            message="Locked-test evaluation is complete",
            validity=self.validity,
        )

    def _custom_search(
        self,
        variant: str,
        temperatures: dict[str, float],
        rho: float,
        use_recall_guardrails: bool,
    ) -> ThresholdSearchResult:
        key = self._ensure_variant(variant)
        bundle = self.predictions[f"{key}:threshold_validation"]
        final_predictions = bundle.logits["final"].argmax(dim=1)
        final_overall = classification_metrics(bundle.labels.numpy(), final_predictions.numpy())
        final_classes = per_class_metrics(bundle.labels.numpy(), final_predictions.numpy(), self.class_names)
        macro_floor = max(0.0, final_overall["macro_f1"] - self.config.routing.macro_f1_tolerance)
        recall_floors: list[float] = []
        for _, row in final_classes.iterrows():
            if not use_recall_guardrails:
                recall_floors.append(0.0)
                continue
            tolerance = (
                self.config.routing.battery_recall_tolerance
                if row["class_name"] == "battery"
                else self.config.routing.recall_tolerance
            )
            recall_floors.append(max(0.0, float(row["recall"]) - tolerance))
        exit_flops = [self.flops[key][name] for name in ("exit1", "exit2", "final")]
        return search_thresholds(
            bundle.logits,
            bundle.labels,
            temperatures,
            self.class_names,
            self.config.routing.base_thresholds,
            self.config.routing.lambdas,
            rho,
            exit_flops,
            macro_floor,
            recall_floors,
            "threshold_validation",
            self.config.routing.min_threshold,
            self.config.routing.max_threshold,
        )

    @tracked_stage("ablations")
    def run_ablations(self, variant: str = "self_distill") -> StageResult:
        frozen_path = self._directories()["results"] / f"frozen_{variant}.json"
        if not frozen_path.exists():
            raise RuntimeError("A frozen manifest is required before test-set ablations")
        key = self._ensure_variant(variant)
        if variant not in self.searches or variant not in self.calibrations:
            raise RuntimeError("Freeze the proposed variant before running ablations")
        test_bundle = self.predictions[f"{key}:test"]
        search = self.searches[variant]
        temperatures = self.calibrations[variant].temperatures
        best = search.best_row
        proposed = route_cached_logits(test_bundle.logits, temperatures, search.thresholds)
        no_calibration_search = self._custom_search(
            variant,
            {"exit1": 1.0, "exit2": 1.0, "final": 1.0},
            self.config.routing.rho,
            True,
        )
        no_calibration = route_cached_logits(
            test_bundle.logits,
            {"exit1": 1.0, "exit2": 1.0, "final": 1.0},
            no_calibration_search.thresholds,
        )
        lambda_zero_thresholds = make_class_thresholds(
            search.difficulty,
            {"exit1": float(best["tau_exit1"]), "exit2": float(best["tau_exit2"])},
            0.0,
            self.config.routing.min_threshold,
            self.config.routing.max_threshold,
        )
        lambda_zero = route_cached_logits(test_bundle.logits, temperatures, lambda_zero_thresholds)
        exit2_only_thresholds = {
            "exit1": torch.full((len(self.class_names),), 1.1),
            "exit2": search.thresholds["exit2"],
        }
        exit2_only = route_cached_logits(test_bundle.logits, temperatures, exit2_only_thresholds)
        rho_zero_search = self._custom_search(variant, temperatures, 0.0, True)
        rho_zero = route_cached_logits(test_bundle.logits, temperatures, rho_zero_search.thresholds)
        no_guard_search = self._custom_search(
            variant,
            temperatures,
            self.config.routing.rho,
            False,
        )
        no_guard = route_cached_logits(test_bundle.logits, temperatures, no_guard_search.thresholds)

        if variant == "ce":
            no_distillation = proposed
        else:
            no_kd_key = self._ensure_variant("ce")
            if "ce" not in self.calibrations:
                self.calibrate_variant("ce")
            if "ce" not in self.searches:
                self.search_variant("ce")
            no_distillation = route_cached_logits(
                self.predictions[f"{no_kd_key}:test"].logits,
                self.calibrations["ce"].temperatures,
                self.searches["ce"].thresholds,
            )

        routed_methods = {
            "Proposed": proposed,
            "No temperature calibration": no_calibration,
            "No class awareness (lambda=0)": lambda_zero,
            "No self-distillation": no_distillation,
            "Exit 2 only": exit2_only,
            "No difficulty shrinkage (rho=0)": rho_zero,
            "No recall guardrails": no_guard,
        }
        exit_flops = [self.flops[key][name] for name in ("exit1", "exit2", "final")]
        rows = []
        for method, routed in routed_methods.items():
            metrics = classification_metrics(test_bundle.labels.numpy(), routed.predictions.numpy())
            counts = torch.bincount(routed.exit_indices, minlength=3).numpy() / len(routed.exit_indices)
            rows.append(
                {
                    "method": method,
                    **metrics,
                    "average_flops": weighted_dynamic_flops(routed.exit_indices, exit_flops),
                    "exit1_share": counts[0],
                    "exit2_share": counts[1],
                    "final_share": counts[2],
                }
            )
        table = pd.DataFrame(rows)
        path = self._directories()["results"] / f"ablations_{variant}.csv"
        _write_csv_atomic(path, table)
        return StageResult(
            "ablations",
            tables={"ablations": table},
            artifacts={"ablations": path},
            message="Ablations reuse frozen test labels only for final reporting",
            validity=self.validity,
        )

    @tracked_stage("profiling")
    def profile_variant(self, variant: str = "self_distill") -> StageResult:
        key = self._ensure_variant(variant)
        if variant not in self.searches:
            self.search_variant(variant)
        if variant not in self.calibrations:
            self.calibrate_variant(variant)
        model = self.models[key].to(self.device).eval()
        images, _labels, _sample_ids = next(iter(self.loaders["test"]))
        inputs = images[:1].to(self.device)
        temperatures = self.calibrations[variant].temperatures
        thresholds = self.searches[variant].thresholds

        def full_operation(values: torch.Tensor):
            return model.forward_all(values)

        def dynamic_operation(values: torch.Tensor):
            return model.forward_dynamic(values, temperatures, thresholds)

        full_latency = benchmark_latency(
            full_operation,
            inputs,
            self.device,
            self.config.profiling.warmup_runs,
            self.config.profiling.latency_runs,
        )
        dynamic_latency = benchmark_latency(
            dynamic_operation,
            inputs,
            self.device,
            self.config.profiling.warmup_runs,
            self.config.profiling.latency_runs,
        )
        full_latency["method"] = "Full ResNet-18"
        dynamic_latency["method"] = "Proposed dynamic"
        latency_table = pd.DataFrame([full_latency, dynamic_latency])
        _, energy = track_estimated_energy(
            lambda: benchmark_latency(
                dynamic_operation,
                inputs,
                self.device,
                self.config.profiling.warmup_runs,
                self.config.profiling.latency_runs,
            ),
            self._directories()["logs"] / "energy",
            self.config.profiling.energy_backend,
        )
        energy_table = pd.DataFrame(
            [
                {
                    "status": energy.status,
                    "method": energy.method,
                    "energy_kwh": energy.energy_kwh,
                    "energy_j": energy.energy_j,
                    "duration_seconds": energy.duration_seconds,
                    "co2e_g": compute_co2e(
                        energy.energy_kwh,
                        self.config.profiling.carbon_intensity_g_per_kwh,
                    ),
                    "detail": energy.detail,
                }
            ],
            dtype=object,
        )
        flops_table = pd.DataFrame(
            [
                {"exit": exit_name, "flops": value}
                for exit_name, value in self.flops[key].items()
            ]
        )
        if f"{key}:test" in self.predictions:
            bundle = self.predictions[f"{key}:test"]
            routed = route_cached_logits(bundle.logits, temperatures, thresholds)
            flops_table["mean_dynamic_flops"] = weighted_dynamic_flops(
                routed.exit_indices,
                [self.flops[key][name] for name in ("exit1", "exit2", "final")],
            )
        results_root = self._directories()["results"]
        paths = {
            "flops": results_root / f"flops_{variant}.csv",
            "latency": results_root / f"latency_{variant}.csv",
            "energy": results_root / f"energy_{variant}.csv",
        }
        _write_csv_atomic(paths["flops"], flops_table)
        _write_csv_atomic(paths["latency"], latency_table)
        _write_csv_atomic(paths["energy"], energy_table)
        return StageResult(
            "profiling",
            tables={"flops": flops_table, "latency": latency_table, "energy": energy_table},
            artifacts=paths,
            message="FLOPs, latency, and energy source are reported separately",
            validity=self.validity,
        )


def run_seed_experiment(
    runner: ExperimentRunner,
    variant: str = "self_distill",
) -> dict[str, StageResult]:
    stages: dict[str, StageResult] = {}
    stages["setup"] = runner.setup()
    stages["audit"] = runner.audit_data()
    stages["splits"] = runner.prepare_splits()
    stages["static_training"] = runner.train_static_baselines()
    stages["early_training"] = runner.train_early_exit_models()
    stages["calibration"] = runner.calibrate_variant(variant)
    stages["threshold_search"] = runner.search_variant(variant)
    stages["freeze"] = runner.freeze_variant(variant)
    stages["locked_test"] = runner.evaluate_locked_test(variant)
    stages["ablations"] = runner.run_ablations(variant)
    stages["profiling"] = runner.profile_variant(variant)
    return stages
