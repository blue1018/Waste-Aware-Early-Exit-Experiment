from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from waste_early_exit.config import load_config
from waste_early_exit.reproducibility import environment_snapshot, resolve_device, seed_everything


def write_config(path: Path, mode: str = "smoke") -> Path:
    payload = {
        "mode": mode,
        "seed": 42,
        "device": "auto",
        "paths": {
            "dataset_root": "garbage_classification",
            "paper_root": "../paper",
            "artifact_root": "artifacts",
        },
        "data": {"num_classes": 12, "image_size": 224},
        "training": {"epochs": 2, "batch_size": 16},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_load_config_resolves_project_paths(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "smoke.yaml")

    config = load_config(config_path, project_root=tmp_path)

    assert config.mode == "smoke"
    assert config.paths.dataset_root == (tmp_path / "garbage_classification").resolve()
    assert config.paths.paper_root == (tmp_path.parent / "paper").resolve()
    assert config.paths.artifact_root == (tmp_path / "artifacts").resolve()
    assert config.data.num_classes == 12


def test_environment_dataset_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_config(tmp_path / "smoke.yaml")
    external_dataset = tmp_path / "external-data"
    monkeypatch.setenv("WEE_DATASET_ROOT", str(external_dataset))

    config = load_config(config_path, project_root=tmp_path)

    assert config.paths.dataset_root == external_dataset.resolve()


def test_runtime_overrides_are_applied(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "smoke.yaml")

    config = load_config(
        config_path,
        project_root=tmp_path,
        overrides={"mode": "full", "training.epochs": 5},
    )

    assert config.mode == "full"
    assert config.training.epochs == 5


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "bad.yaml", mode="demo")

    with pytest.raises(ValueError, match="mode"):
        load_config(config_path, project_root=tmp_path)


def test_requested_unavailable_mps_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_seed_everything_repeats_numpy_and_torch_values() -> None:
    seed_everything(123)
    first_numpy = np.random.random(3)
    first_torch = torch.rand(3)
    seed_everything(123)

    assert np.allclose(np.random.random(3), first_numpy)
    assert torch.equal(torch.rand(3), first_torch)


def test_environment_snapshot_has_reproducibility_fields() -> None:
    snapshot = environment_snapshot(torch.device("cpu"))

    assert snapshot["device"] == "cpu"
    assert snapshot["python"]
    assert snapshot["torch"]
    assert snapshot["platform"]

