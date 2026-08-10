from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix


sns.set_theme(style="whitegrid", context="notebook")


def _save(figure: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_training_history(history: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="Train")
    axes[0].plot(history["epoch"], history["validation_loss"], marker="o", label="Validation")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_macro_f1"], marker="o", label="Train")
    axes[1].plot(history["epoch"], history["validation_macro_f1"], marker="o", label="Validation")
    axes[1].set(title="Macro F1", xlabel="Epoch", ylabel="Macro F1", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    return _save(figure, output_path)


def plot_confusion_matrix(
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    output_path: str | Path | None = None,
    normalize: bool = True,
) -> plt.Figure:
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        normalize="true" if normalize else None,
    )
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axis)
    axis.set(title="Normalized confusion matrix" if normalize else "Confusion matrix", xlabel="Predicted class", ylabel="True class")
    figure.tight_layout()
    return _save(figure, output_path)


def plot_exit_shares(table: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    frame = table.set_index("method")[["exit1", "exit2", "final"]]
    figure, axis = plt.subplots(figsize=(8, 4))
    frame.plot.bar(stacked=True, color=["#14B8A6", "#3B82F6", "#64748B"], ax=axis)
    axis.set(title="Exit share by method", xlabel="Method", ylabel="Share", ylim=(0, 1))
    axis.legend(title="Exit", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    return _save(figure, output_path)


def plot_pareto(
    table: pd.DataFrame,
    x: str = "average_flops",
    y: str = "macro_f1",
    output_path: str | Path | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=table, x=x, y=y, hue="method", s=90, ax=axis)
    for _, row in table.iterrows():
        axis.annotate(str(row["method"]), (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set(title=f"{y.replace('_', ' ').title()} vs {x.replace('_', ' ').title()}")
    figure.tight_layout()
    return _save(figure, output_path)


def plot_reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    output_path: str | Path | None = None,
    n_bins: int = 15,
    title: str = "Reliability diagram",
) -> plt.Figure:
    probabilities = torch.softmax(logits.detach().cpu(), dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(labels.detach().cpu()).numpy().astype(int)
    observed, predicted = calibration_curve(correct, confidence.numpy(), n_bins=n_bins, strategy="uniform")
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="#64748B", label="Perfect calibration")
    axis.plot(predicted, observed, marker="o", color="#3B82F6", label="Model")
    axis.set(title=title, xlabel="Mean confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    return _save(figure, output_path)


def plot_threshold_heatmap(
    thresholds: pd.DataFrame,
    output_path: str | Path | None = None,
) -> plt.Figure:
    pivot = thresholds.pivot(index="class_name", columns="exit", values="threshold")
    figure, axis = plt.subplots(figsize=(6, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0.55, vmax=0.99, ax=axis)
    axis.set(title="Class-aware exit thresholds", xlabel="Exit", ylabel="Class")
    figure.tight_layout()
    return _save(figure, output_path)


def plot_per_class_metrics(
    table: pd.DataFrame,
    metric: str = "f1",
    output_path: str | Path | None = None,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(data=table, x="class_name", y=metric, hue="method", ax=axis)
    axis.set(title=f"Per-class {metric.upper()}", xlabel="Class", ylabel=metric.upper(), ylim=(0, 1))
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    return _save(figure, output_path)


def plot_qualitative_examples(
    frame: pd.DataFrame,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    exit_indices: Sequence[int],
    class_names: Sequence[str],
    output_path: str | Path | None = None,
    maximum: int = 12,
) -> plt.Figure:
    lookup = frame.drop_duplicates("relative_path").set_index("relative_path")["path"].to_dict()
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    exits_array = np.asarray(exit_indices)
    errors = np.flatnonzero(labels_array != predictions_array).tolist()
    correct = np.flatnonzero(labels_array == predictions_array).tolist()
    selected = (errors + correct)[:maximum]
    if not selected:
        raise ValueError("No samples are available for qualitative examples")
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))
    flat_axes = np.atleast_1d(axes).ravel()
    exit_names = ("Exit 1", "Exit 2", "Final")
    for axis, sample_index in zip(flat_axes, selected):
        sample_id = str(sample_ids[sample_index])
        if sample_id not in lookup:
            raise KeyError(f"Sample path not found in split table: {sample_id}")
        with Image.open(lookup[sample_id]) as image:
            axis.imshow(image.convert("RGB"))
        true_name = class_names[int(labels_array[sample_index])]
        predicted_name = class_names[int(predictions_array[sample_index])]
        status = "correct" if labels_array[sample_index] == predictions_array[sample_index] else "wrong"
        axis.set_title(
            f"{exit_names[int(exits_array[sample_index])]} | {status}\nTrue: {true_name} | Pred: {predicted_name}",
            fontsize=8,
        )
        axis.axis("off")
    for axis in flat_axes[len(selected) :]:
        axis.axis("off")
    figure.suptitle("Early-exit examples", y=1.01)
    figure.tight_layout()
    return _save(figure, output_path)
