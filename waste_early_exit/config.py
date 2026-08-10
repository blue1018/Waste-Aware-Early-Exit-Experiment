from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    dataset_root: Path
    paper_root: Path
    artifact_root: Path


@dataclass(frozen=True)
class DataConfig:
    num_classes: int = 12
    image_size: int = 224
    num_workers: int = 0
    phash_distance: int = 4
    corruption_tolerance: float = 0.01
    smoke_train_per_class: int = 120
    smoke_eval_per_class: int = 40


@dataclass(frozen=True)
class ModelConfig:
    pretrained: bool = True
    dropout: float = 0.2
    exit_names: tuple[str, ...] = ("exit1", "exit2", "final")
    class_weight_min: float = 0.5
    class_weight_max: float = 3.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 2
    static_epochs: int = 1
    batch_size: int = 16
    min_batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 5
    alpha: tuple[float, ...] = (0.2, 0.3, 0.5)
    kd_temperature: float = 2.0
    kd_gamma: float = 0.5


@dataclass(frozen=True)
class RoutingConfig:
    base_thresholds: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    lambdas: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    rho: float = 20.0
    min_threshold: float = 0.55
    max_threshold: float = 0.99
    macro_f1_tolerance: float = 0.01
    recall_tolerance: float = 0.03
    battery_recall_tolerance: float = 0.01


@dataclass(frozen=True)
class ProfilingConfig:
    warmup_runs: int = 10
    latency_runs: int = 100
    carbon_intensity_g_per_kwh: float | None = None
    energy_backend: str = "codecarbon"


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    seed: int
    seeds: tuple[int, ...]
    device: str
    paths: PathConfig
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def normalized(self) -> dict[str, Any]:
        value = dataclasses.asdict(self)
        for key, path in value["paths"].items():
            value["paths"][key] = str(path)
        return value

    def fingerprint(self) -> str:
        payload = json.dumps(self.normalized(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _deep_set(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = mapping
    for key in keys[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[keys[-1]] = value


def _tuple_values(values: Any, cast: type = float) -> tuple[Any, ...]:
    if values is None:
        return ()
    return tuple(cast(value) for value in values)


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_config(
    path: str | Path,
    project_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping")
    for key, value in (overrides or {}).items():
        _deep_set(payload, key, value)

    mode = str(payload.get("mode", "smoke")).lower()
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")

    root = Path(project_root).expanduser().resolve() if project_root else config_path.parent.parent
    path_values = payload.get("paths", {})
    dataset_value = os.environ.get("WEE_DATASET_ROOT", path_values.get("dataset_root", "garbage_classification"))
    paper_value = os.environ.get("WEE_PAPER_ROOT", path_values.get("paper_root", "../paper"))
    artifact_value = os.environ.get("WEE_ARTIFACT_ROOT", path_values.get("artifact_root", "artifacts"))
    paths = PathConfig(
        dataset_root=_resolve_path(dataset_value, root),
        paper_root=_resolve_path(paper_value, root),
        artifact_root=_resolve_path(artifact_value, root),
    )

    data_values = payload.get("data", {})
    model_values = payload.get("model", {})
    training_values = payload.get("training", {})
    routing_values = payload.get("routing", {})
    profiling_values = payload.get("profiling", {})

    data = DataConfig(**{key: value for key, value in data_values.items() if key in DataConfig.__dataclass_fields__})
    model_payload = {key: value for key, value in model_values.items() if key in ModelConfig.__dataclass_fields__}
    if "exit_names" in model_payload:
        model_payload["exit_names"] = _tuple_values(model_payload["exit_names"], str)
    model = ModelConfig(**model_payload)
    training_payload = {
        key: value for key, value in training_values.items() if key in TrainingConfig.__dataclass_fields__
    }
    if "alpha" in training_payload:
        training_payload["alpha"] = _tuple_values(training_payload["alpha"])
    training = TrainingConfig(**training_payload)
    routing_payload = {
        key: value for key, value in routing_values.items() if key in RoutingConfig.__dataclass_fields__
    }
    for key in ("base_thresholds", "lambdas"):
        if key in routing_payload:
            routing_payload[key] = _tuple_values(routing_payload[key])
    routing = RoutingConfig(**routing_payload)
    profiling = ProfilingConfig(
        **{key: value for key, value in profiling_values.items() if key in ProfilingConfig.__dataclass_fields__}
    )
    seed = int(payload.get("seed", 42))
    default_seeds = (seed,) if mode == "smoke" else (42, 123, 2026)
    seeds = tuple(int(value) for value in payload.get("seeds", default_seeds))
    return ExperimentConfig(
        mode=mode,
        seed=seed,
        seeds=seeds,
        device=str(payload.get("device", "auto")).lower(),
        paths=paths,
        data=data,
        model=model,
        training=training,
        routing=routing,
        profiling=profiling,
    )

