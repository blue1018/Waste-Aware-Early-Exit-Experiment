import math

import numpy as np
import torch

from waste_early_exit.metrics import calibration_metrics, classification_metrics, per_class_metrics


def test_calibration_metrics_match_hand_checked_two_class_example() -> None:
    probabilities = torch.tensor([[0.8, 0.2], [0.4, 0.6]], dtype=torch.float64)
    logits = probabilities.log()
    labels = torch.tensor([0, 1])

    metrics = calibration_metrics(logits, labels, n_bins=2)

    assert math.isclose(metrics["nll"], -(math.log(0.8) + math.log(0.6)) / 2, rel_tol=1e-6)
    assert math.isclose(metrics["brier"], 0.2, rel_tol=1e-6)
    assert math.isclose(metrics["ece"], 0.3, rel_tol=1e-6)


def test_classification_and_per_class_metrics_are_consistent() -> None:
    labels = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 1, 1])

    overall = classification_metrics(labels, predictions)
    classes = per_class_metrics(labels, predictions, ["zero", "one"])

    assert overall["accuracy"] == 0.75
    assert math.isclose(overall["macro_recall"], 0.75)
    assert classes.loc[classes["class_name"] == "zero", "recall"].iloc[0] == 0.5
    assert classes["support"].sum() == 4

