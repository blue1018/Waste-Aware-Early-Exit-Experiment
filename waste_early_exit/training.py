from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .losses import multi_exit_loss
from .reproducibility import synchronize_device

if TYPE_CHECKING:
    from .progress import ProgressReporter


@dataclass(frozen=True)
class PredictionBundle:
    logits: dict[str, torch.Tensor]
    labels: torch.Tensor
    sample_ids: list[str]


@dataclass(frozen=True)
class TrainingResult:
    history: pd.DataFrame
    best_metric: float
    best_epoch: int
    checkpoint_path: Path
    effective_batch_size: int
    estimated_total_seconds: float


def save_checkpoint_atomic(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    best_metric: float,
    effective_batch_size: int,
    scheduler: Any | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "effective_batch_size": int(effective_batch_size),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {key: payload[key] for key in ("epoch", "best_metric", "effective_batch_size")}


def _clone_loader(loader: DataLoader[Any], batch_size: int, shuffle: bool) -> DataLoader[Any]:
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        pin_memory=False,
        drop_last=False,
        persistent_workers=loader.num_workers > 0,
    )


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _forward_loss(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None,
    config: ExperimentConfig,
    use_distillation: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(images)
    if isinstance(outputs, dict):
        breakdown = multi_exit_loss(
            outputs,
            labels,
            class_weights,
            config.training.alpha,
            config.training.kd_temperature,
            config.training.kd_gamma,
            use_distillation,
        )
        return breakdown.total, outputs["final"]
    return F.cross_entropy(outputs, labels, weight=class_weights), outputs


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    config: ExperimentConfig,
    class_weights: torch.Tensor | None,
    optimizer: torch.optim.Optimizer | None,
    use_distillation: bool,
    progress: ProgressReporter | None = None,
    progress_seed: int | None = None,
    progress_label: str = "model",
    epoch: int = 1,
    total_epochs: int = 1,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    context = torch.enable_grad() if training else torch.inference_mode()
    phase = "train" if training else "validation"
    iterable = (
        progress.batch(
            loader,
            int(progress_seed),
            phase,
            model=progress_label,
            epoch=epoch,
            total_epochs=total_epochs,
        )
        if progress is not None and progress_seed is not None
        else loader
    )
    started = time.perf_counter()
    processed_images = 0
    with context:
        for batch_index, (images, labels, _sample_ids) in enumerate(iterable, start=1):
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, final_logits = _forward_loss(
                model, images, labels, class_weights, config, use_distillation
            )
            if training:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            labels_all.extend(labels.detach().cpu().tolist())
            predictions_all.extend(final_logits.detach().argmax(dim=1).cpu().tolist())
            processed_images += int(labels.shape[0])
            if progress is not None:
                elapsed = max(time.perf_counter() - started, 1e-9)
                progress.update(
                    batch=batch_index,
                    total_batches=len(loader),
                    processed_images=processed_images,
                    images_per_second=processed_images / elapsed,
                    mean_loss=float(np.mean(losses)),
                )
    macro_f1 = f1_score(labels_all, predictions_all, average="macro", zero_division=0)
    return float(np.mean(losses)), float(macro_f1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    config: ExperimentConfig,
    device: torch.device,
    checkpoint_path: str | Path,
    class_weights: torch.Tensor | None = None,
    use_distillation: bool = False,
    epochs: int | None = None,
    progress: ProgressReporter | None = None,
    progress_seed: int | None = None,
    progress_label: str = "model",
) -> TrainingResult:
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    total_epochs = int(epochs or config.training.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    effective_batch_size = train_loader.batch_size or config.training.batch_size
    best_metric = -1.0
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    seed = int(progress_seed if progress_seed is not None else config.seed)
    if progress is not None:
        progress.event(
            "model_started",
            seed,
            stage="training",
            model=progress_label,
            total_epochs=total_epochs,
            batch_size=effective_batch_size,
            device=str(device),
        )
    epoch = 0
    while epoch < total_epochs:
        epoch_model = copy.deepcopy(model.state_dict())
        epoch_optimizer = copy.deepcopy(optimizer.state_dict())
        try:
            train_loss, train_f1 = _run_epoch(
                model,
                train_loader,
                device,
                config,
                class_weights,
                optimizer,
                use_distillation,
                progress,
                seed,
                progress_label,
                epoch + 1,
                total_epochs,
            )
            validation_loss, validation_f1 = _run_epoch(
                model,
                validation_loader,
                device,
                config,
                class_weights,
                None,
                use_distillation,
                progress,
                seed,
                progress_label,
                epoch + 1,
                total_epochs,
            )
        except RuntimeError as error:
            if device.type != "mps" or not _is_oom(error):
                raise
            next_batch_size = effective_batch_size // 2
            if next_batch_size < config.training.min_batch_size:
                raise RuntimeError("MPS ran out of memory at the minimum batch size") from error
            model.load_state_dict(epoch_model)
            optimizer.load_state_dict(epoch_optimizer)
            previous_batch_size = effective_batch_size
            effective_batch_size = next_batch_size
            train_loader = _clone_loader(train_loader, effective_batch_size, shuffle=True)
            validation_loader = _clone_loader(validation_loader, effective_batch_size, shuffle=False)
            torch.mps.empty_cache()
            if progress is not None:
                progress.event(
                    "batch_size_reduced",
                    seed,
                    stage="training",
                    model=progress_label,
                    epoch=epoch + 1,
                    previous_batch_size=previous_batch_size,
                    batch_size=effective_batch_size,
                    reason="mps_out_of_memory",
                )
            continue
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "validation_loss": validation_loss,
                "validation_macro_f1": validation_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": effective_batch_size,
            }
        )
        if validation_f1 > best_metric:
            best_metric = validation_f1
            best_epoch = epoch + 1
            stale_epochs = 0
            save_checkpoint_atomic(
                checkpoint_path,
                model,
                optimizer,
                best_epoch,
                best_metric,
                effective_batch_size,
                scheduler,
            )
        else:
            stale_epochs += 1
        epoch += 1
        if progress is not None:
            elapsed = time.perf_counter() - started
            mean_epoch_seconds = elapsed / max(epoch, 1)
            progress.event(
                "epoch_complete",
                seed,
                stage="training",
                model=progress_label,
                epoch=epoch,
                total_epochs=total_epochs,
                train_loss=train_loss,
                train_macro_f1=train_f1,
                validation_loss=validation_loss,
                validation_macro_f1=validation_f1,
                best_validation_macro_f1=best_metric,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                patience=config.training.patience,
                learning_rate=optimizer.param_groups[0]["lr"],
                batch_size=effective_batch_size,
                elapsed_seconds=elapsed,
                eta_seconds=max(0.0, mean_epoch_seconds * (total_epochs - epoch)),
            )
        if stale_epochs >= config.training.patience:
            if progress is not None:
                progress.event(
                    "early_stopping",
                    seed,
                    stage="training",
                    model=progress_label,
                    best_epoch=best_epoch,
                    best_validation_macro_f1=best_metric,
                    patience=config.training.patience,
                    completed_epochs=epoch,
                )
            break
    synchronize_device(device)
    elapsed = time.perf_counter() - started
    completed_epochs = max(len(history), 1)
    estimated_total = elapsed / completed_epochs * total_epochs
    load_checkpoint(checkpoint_path, model, map_location=device)
    if progress is not None:
        progress.event(
            "model_complete",
            seed,
            stage="training",
            model=progress_label,
            completed_epochs=len(history),
            best_epoch=best_epoch,
            best_validation_macro_f1=best_metric,
            batch_size=effective_batch_size,
            elapsed_seconds=elapsed,
        )
    return TrainingResult(
        history=pd.DataFrame(history),
        best_metric=best_metric,
        best_epoch=best_epoch,
        checkpoint_path=Path(checkpoint_path),
        effective_batch_size=effective_batch_size,
        estimated_total_seconds=float(estimated_total),
    )


def predict_logits(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    progress: ProgressReporter | None = None,
    progress_seed: int | None = None,
    progress_label: str = "prediction",
) -> PredictionBundle:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {}
    labels_all: list[torch.Tensor] = []
    sample_ids: list[str] = []
    iterable = (
        progress.batch(
            loader,
            int(progress_seed),
            "prediction",
            model=progress_label,
        )
        if progress is not None and progress_seed is not None
        else loader
    )
    with torch.inference_mode():
        for images, labels, batch_ids in iterable:
            outputs = model(images.to(device))
            if not isinstance(outputs, dict):
                outputs = {"final": outputs}
            for name, logits in outputs.items():
                collected.setdefault(name, []).append(logits.detach().cpu())
            labels_all.append(labels.detach().cpu())
            sample_ids.extend(str(value) for value in batch_ids)
    return PredictionBundle(
        logits={name: torch.cat(values) for name, values in collected.items()},
        labels=torch.cat(labels_all),
        sample_ids=sample_ids,
    )
