from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from .metrics import classification_metrics, per_class_metrics


@dataclass(frozen=True)
class RoutingResult:
    predictions: torch.Tensor
    exit_indices: torch.Tensor
    confidences: torch.Tensor
    selected_logits: torch.Tensor


@dataclass(frozen=True)
class ThresholdSearchResult:
    best_row: dict[str, float | bool]
    table: pd.DataFrame
    thresholds: dict[str, torch.Tensor]
    difficulty: pd.DataFrame


def shrink_difficulty(
    difficulty: Sequence[float],
    support: Sequence[int],
    rho: float,
) -> np.ndarray:
    difficulty_values = np.asarray(difficulty, dtype=np.float64)
    support_values = np.asarray(support, dtype=np.float64)
    mean = float(difficulty_values.mean())
    return (support_values * difficulty_values + float(rho) * mean) / (support_values + float(rho))


def estimate_class_difficulty(
    predictions_by_exit: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    num_classes: int,
    rho: float,
) -> pd.DataFrame:
    targets = labels.detach().cpu().numpy()
    rows: list[dict[str, float | int | str]] = []
    for exit_name in ("exit1", "exit2"):
        predictions = predictions_by_exit[exit_name].detach().cpu().numpy()
        _, _, f1, support = precision_recall_fscore_support(
            targets,
            predictions,
            labels=list(range(num_classes)),
            average=None,
            zero_division=0,
        )
        difficulty = 1.0 - f1
        shrunk = shrink_difficulty(difficulty, support, rho)
        mean = float(shrunk.mean())
        for class_index in range(num_classes):
            rows.append(
                {
                    "exit": exit_name,
                    "class_index": class_index,
                    "support": int(support[class_index]),
                    "f1": float(f1[class_index]),
                    "difficulty": float(difficulty[class_index]),
                    "shrunk_difficulty": float(shrunk[class_index]),
                    "mean_difficulty": mean,
                }
            )
    return pd.DataFrame(rows)


def make_class_thresholds(
    difficulty: pd.DataFrame,
    base_thresholds: Mapping[str, float],
    lambda_value: float,
    minimum: float,
    maximum: float,
) -> dict[str, torch.Tensor]:
    thresholds: dict[str, torch.Tensor] = {}
    for exit_name in ("exit1", "exit2"):
        frame = difficulty.loc[difficulty["exit"] == exit_name].sort_values("class_index")
        centered = frame["shrunk_difficulty"].to_numpy() - frame["mean_difficulty"].to_numpy()
        values = np.clip(float(base_thresholds[exit_name]) + float(lambda_value) * centered, minimum, maximum)
        thresholds[exit_name] = torch.tensor(values, dtype=torch.float32)
    return thresholds


def route_cached_logits(
    logits: Mapping[str, torch.Tensor],
    temperatures: Mapping[str, float],
    thresholds: Mapping[str, torch.Tensor],
) -> RoutingResult:
    names = ("exit1", "exit2", "final")
    count = logits["final"].shape[0]
    classes = logits["final"].shape[1]
    selected_logits = torch.zeros(count, classes, dtype=logits["final"].dtype)
    exit_indices = torch.full((count,), 2, dtype=torch.long)
    confidences = torch.zeros(count, dtype=torch.float32)
    active = torch.ones(count, dtype=torch.bool)
    for exit_index, exit_name in enumerate(("exit1", "exit2")):
        probabilities = torch.softmax(logits[exit_name].detach().cpu() / float(temperatures[exit_name]), dim=1)
        confidence, prediction = probabilities.max(dim=1)
        threshold = thresholds[exit_name].detach().cpu()[prediction]
        leave = active & (confidence >= threshold)
        selected_logits[leave] = logits[exit_name].detach().cpu()[leave]
        exit_indices[leave] = exit_index
        confidences[leave] = confidence[leave].to(dtype=torch.float32)
        active = active & ~leave
    final_probabilities = torch.softmax(logits["final"].detach().cpu() / float(temperatures["final"]), dim=1)
    final_confidence, _ = final_probabilities.max(dim=1)
    selected_logits[active] = logits["final"].detach().cpu()[active]
    confidences[active] = final_confidence[active].to(dtype=torch.float32)
    predictions = selected_logits.argmax(dim=1)
    return RoutingResult(predictions, exit_indices, confidences, selected_logits)


def search_thresholds(
    logits: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    temperatures: Mapping[str, float],
    class_names: Sequence[str],
    base_threshold_grid: Sequence[float],
    lambda_grid: Sequence[float],
    rho: float,
    exit_flops: Sequence[float],
    macro_f1_floor: float,
    recall_floors: Sequence[float],
    split_name: str,
    minimum: float = 0.55,
    maximum: float = 0.99,
) -> ThresholdSearchResult:
    if split_name != "threshold_validation":
        raise ValueError("Threshold search must use the threshold_validation split")
    calibrated_predictions = {
        name: torch.softmax(values.detach().cpu() / float(temperatures[name]), dim=1).argmax(dim=1)
        for name, values in logits.items()
    }
    difficulty = estimate_class_difficulty(
        {name: calibrated_predictions[name] for name in ("exit1", "exit2")},
        labels,
        len(class_names),
        rho,
    )
    rows: list[dict[str, float | bool]] = []
    flops = np.asarray(exit_flops, dtype=np.float64)
    recall_floor_values = np.asarray(recall_floors, dtype=np.float64)
    for tau1, tau2, lambda_value in itertools.product(
        base_threshold_grid,
        base_threshold_grid,
        lambda_grid,
    ):
        thresholds = make_class_thresholds(
            difficulty,
            {"exit1": float(tau1), "exit2": float(tau2)},
            float(lambda_value),
            minimum,
            maximum,
        )
        routing = route_cached_logits(logits, temperatures, thresholds)
        overall = classification_metrics(labels.numpy(), routing.predictions.numpy())
        classes = per_class_metrics(labels.numpy(), routing.predictions.numpy(), class_names)
        feasible = bool(
            overall["macro_f1"] >= macro_f1_floor
            and np.all(classes["recall"].to_numpy() >= recall_floor_values)
        )
        exit_counts = torch.bincount(routing.exit_indices, minlength=3).numpy()
        rows.append(
            {
                "tau_exit1": float(tau1),
                "tau_exit2": float(tau2),
                "lambda": float(lambda_value),
                "macro_f1": float(overall["macro_f1"]),
                "average_flops": float(flops[routing.exit_indices.numpy()].mean()),
                "exit1_share": float(exit_counts[0] / len(labels)),
                "exit2_share": float(exit_counts[1] / len(labels)),
                "final_share": float(exit_counts[2] / len(labels)),
                "minimum_recall": float(classes["recall"].min()),
                "feasible": feasible,
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["feasible", "average_flops", "macro_f1", "tau_exit1", "tau_exit2"],
        ascending=[False, True, False, True, True],
    ).reset_index(drop=True)
    feasible_rows = table.loc[table["feasible"]]
    if feasible_rows.empty:
        raise RuntimeError("No threshold setting satisfies the F1 and recall constraints")
    best_row = feasible_rows.iloc[0].to_dict()
    best_thresholds = make_class_thresholds(
        difficulty,
        {"exit1": float(best_row["tau_exit1"]), "exit2": float(best_row["tau_exit2"])},
        float(best_row["lambda"]),
        minimum,
        maximum,
    )
    return ThresholdSearchResult(best_row, table, best_thresholds, difficulty)


def oracle_route(logits: Mapping[str, torch.Tensor], labels: torch.Tensor) -> RoutingResult:
    predictions = {name: values.argmax(dim=1) for name, values in logits.items()}
    thresholds = {"exit1": torch.ones(logits["final"].shape[1]), "exit2": torch.ones(logits["final"].shape[1])}
    result = route_cached_logits(logits, {"exit1": 1.0, "exit2": 1.0, "final": 1.0}, thresholds)
    selected = result.selected_logits.clone()
    exits = result.exit_indices.clone()
    for index in range(len(labels)):
        if predictions["exit1"][index] == labels[index]:
            selected[index] = logits["exit1"][index]
            exits[index] = 0
        elif predictions["exit2"][index] == labels[index]:
            selected[index] = logits["exit2"][index]
            exits[index] = 1
        else:
            selected[index] = logits["final"][index]
            exits[index] = 2
    selected_predictions = selected.argmax(dim=1)
    confidences = torch.softmax(selected, dim=1).max(dim=1).values
    return RoutingResult(selected_predictions, exits, confidences, selected)

