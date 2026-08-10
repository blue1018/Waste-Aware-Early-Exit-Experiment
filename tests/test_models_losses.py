import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from waste_early_exit.config import load_config
from waste_early_exit.losses import compute_class_weights, multi_exit_loss
from waste_early_exit.models import EarlyExitResNet18, build_static_model
from waste_early_exit.progress import ProgressReporter
from waste_early_exit.training import load_checkpoint, save_checkpoint_atomic, train_model


@pytest.mark.parametrize("name", ["mobilenet_v3_small", "efficientnet_b0", "resnet18"])
def test_static_models_emit_requested_class_count(name: str) -> None:
    model = build_static_model(name, num_classes=12, pretrained=False).eval()

    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 64, 64))

    assert tuple(logits.shape) == (2, 12)


def test_early_exit_model_emits_all_three_logits() -> None:
    model = EarlyExitResNet18(num_classes=12, pretrained=False).eval()

    with torch.inference_mode():
        outputs = model.forward_all(torch.randn(2, 3, 64, 64))

    assert set(outputs) == {"exit1", "exit2", "final"}
    assert all(tuple(logits.shape) == (2, 12) for logits in outputs.values())


def test_dynamic_routing_can_send_every_sample_to_first_or_final_exit() -> None:
    model = EarlyExitResNet18(num_classes=12, pretrained=False).eval()
    inputs = torch.randn(3, 3, 64, 64)
    temperatures = {"exit1": 1.0, "exit2": 1.0, "final": 1.0}

    with torch.inference_mode():
        first = model.forward_dynamic(
            inputs,
            temperatures,
            {"exit1": torch.zeros(12), "exit2": torch.zeros(12)},
        )
        final = model.forward_dynamic(
            inputs,
            temperatures,
            {"exit1": torch.full((12,), 1.1), "exit2": torch.full((12,), 1.1)},
        )

    assert torch.equal(first.exit_indices, torch.zeros(3, dtype=torch.long))
    assert torch.equal(final.exit_indices, torch.full((3,), 2, dtype=torch.long))
    assert tuple(first.logits.shape) == (3, 12)
    assert tuple(final.logits.shape) == (3, 12)


def test_class_weights_match_hand_checked_square_root_weights() -> None:
    weights = compute_class_weights(torch.tensor([1, 4]), minimum=0.1, maximum=5.0)

    assert torch.allclose(weights, torch.tensor([4 / 3, 2 / 3]), atol=1e-5)


def test_distillation_does_not_add_gradient_to_final_teacher() -> None:
    targets = torch.tensor([0, 1, 2])
    base = {
        "exit1": torch.tensor([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]),
        "exit2": torch.tensor([[1.5, 0.2, -0.5], [0.1, 1.5, -0.5], [0.1, -0.5, 1.5]]),
        "final": torch.tensor([[2.5, 0.1, -0.5], [0.1, 2.5, -0.5], [0.1, -0.5, 2.5]]),
    }

    def gradients(use_distillation: bool) -> dict[str, torch.Tensor]:
        logits = {name: value.clone().requires_grad_(True) for name, value in base.items()}
        breakdown = multi_exit_loss(
            logits,
            targets,
            class_weights=None,
            alpha=(0.2, 0.3, 0.5),
            kd_temperature=2.0,
            kd_gamma=0.5,
            use_distillation=use_distillation,
        )
        breakdown.total.backward()
        return {name: value.grad.detach().clone() for name, value in logits.items()}

    without_kd = gradients(False)
    with_kd = gradients(True)

    assert torch.allclose(without_kd["final"], with_kd["final"], atol=1e-6)
    assert not torch.allclose(without_kd["exit1"], with_kd["exit1"])


def test_checkpoint_round_trip_restores_model_and_metadata(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint_atomic(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        best_metric=0.7,
        effective_batch_size=8,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)

    metadata = load_checkpoint(checkpoint_path, model, optimizer, map_location="cpu")

    assert metadata["epoch"] == 3
    assert metadata["best_metric"] == 0.7
    assert metadata["effective_batch_size"] == 8
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name])


def training_fixture(tmp_path: Path, epochs: int, patience: int, learning_rate: float = 0.01):
    project_root = Path(__file__).parents[1]
    config = load_config(
        project_root / "configs" / "smoke.yaml",
        project_root=project_root,
        overrides={
            "device": "cpu",
            "training.epochs": epochs,
            "training.patience": patience,
            "training.learning_rate": learning_rate,
            "training.batch_size": 4,
        },
    )
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(12, 1, 2, 2, generator=generator)
    labels = torch.tensor([0, 1] * 6)
    sample_ids = torch.arange(12)
    loader = DataLoader(TensorDataset(images, labels, sample_ids), batch_size=4, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    aggregate = tmp_path / "aggregate.log"
    reporter = ProgressReporter(
        ("42:training",), aggregate, {42: tmp_path / "seed.log"}, display=False
    )
    return config, loader, model, reporter, aggregate


def test_train_model_reports_epoch_metrics_and_eta(tmp_path: Path) -> None:
    config, loader, model, reporter, aggregate = training_fixture(tmp_path, epochs=1, patience=1)

    result = train_model(
        model,
        loader,
        loader,
        config,
        torch.device("cpu"),
        tmp_path / "model.pt",
        epochs=1,
        progress=reporter,
        progress_seed=42,
        progress_label="tiny",
    )
    reporter.close("complete")
    records = [json.loads(line) for line in aggregate.read_text(encoding="utf-8").splitlines()]
    epoch = next(record for record in records if record["kind"] == "epoch_complete")

    assert len(result.history) == 1
    assert epoch["epoch"] == 1
    assert epoch["model"] == "tiny"
    assert 0.0 <= epoch["validation_macro_f1"] <= 1.0
    assert epoch["eta_seconds"] >= 0.0
    assert epoch["total_epochs"] == 1


def test_train_model_reports_early_stopping(tmp_path: Path) -> None:
    config, loader, model, reporter, aggregate = training_fixture(
        tmp_path, epochs=3, patience=1, learning_rate=0.0
    )

    result = train_model(
        model,
        loader,
        loader,
        config,
        torch.device("cpu"),
        tmp_path / "model.pt",
        epochs=3,
        progress=reporter,
        progress_seed=42,
        progress_label="tiny",
    )
    reporter.close("complete")
    records = [json.loads(line) for line in aggregate.read_text(encoding="utf-8").splitlines()]

    assert len(result.history) == 2
    early = next(record for record in records if record["kind"] == "early_stopping")
    assert early["best_epoch"] == 1
    assert early["completed_epochs"] == 2
