from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def per_class_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> pd.DataFrame:
    indices = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=indices,
        average=None,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "class_index": indices,
            "class_name": list(class_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )


def calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> dict[str, float]:
    values = logits.detach().to(dtype=torch.float64, device="cpu")
    targets = labels.detach().to(dtype=torch.long, device="cpu")
    probabilities = torch.softmax(values, dim=1)
    nll = F.cross_entropy(values, targets).item()
    one_hot = F.one_hot(targets, num_classes=values.shape[1]).to(dtype=torch.float64)
    brier = torch.sum((probabilities - one_hot) ** 2, dim=1).mean().item()
    confidences, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets).to(dtype=torch.float64)
    boundaries = torch.linspace(0, 1, n_bins + 1, dtype=torch.float64)
    ece = torch.zeros((), dtype=torch.float64)
    for index in range(n_bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        if mask.any():
            weight = mask.to(dtype=torch.float64).mean()
            ece = ece + weight * torch.abs(correct[mask].mean() - confidences[mask].mean())
    return {"nll": float(nll), "brier": float(brier), "ece": float(ece.item())}


def paired_bootstrap(
    labels: Sequence[int],
    predictions_a: Sequence[int],
    predictions_b: Sequence[int],
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    labels_array = np.asarray(labels)
    first = np.asarray(predictions_a)
    second = np.asarray(predictions_b)
    if not (len(labels_array) == len(first) == len(second)):
        raise ValueError("Bootstrap inputs must have the same length")
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        sample = rng.integers(0, len(labels_array), size=len(labels_array))
        first_f1 = classification_metrics(labels_array[sample], first[sample])["macro_f1"]
        second_f1 = classification_metrics(labels_array[sample], second[sample])["macro_f1"]
        differences[index] = first_f1 - second_f1
    return {
        "difference": float(classification_metrics(labels_array, first)["macro_f1"] - classification_metrics(labels_array, second)["macro_f1"]),
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "resamples": int(n_resamples),
    }

