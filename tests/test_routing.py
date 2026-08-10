import math

import numpy as np
import pandas as pd
import pytest
import torch

from waste_early_exit.routing import (
    estimate_class_difficulty,
    make_class_thresholds,
    route_cached_logits,
    search_thresholds,
    shrink_difficulty,
)


def test_difficulty_shrinkage_moves_small_classes_more_toward_mean() -> None:
    difficulty = np.array([0.1, 0.9])
    support = np.array([100, 2])

    shrunk = shrink_difficulty(difficulty, support, rho=20)

    mean = difficulty.mean()
    assert abs(shrunk[1] - mean) < abs(shrunk[0] - mean)
    assert abs(shrunk[1] - difficulty[1]) > abs(shrunk[0] - difficulty[0])


def test_class_thresholds_match_hand_checked_centered_difficulty() -> None:
    difficulty = pd.DataFrame(
        {
            "exit": ["exit1", "exit1", "exit2", "exit2"],
            "class_index": [0, 1, 0, 1],
            "shrunk_difficulty": [0.0, 1.0, 0.25, 0.75],
            "mean_difficulty": [0.5, 0.5, 0.5, 0.5],
        }
    )

    thresholds = make_class_thresholds(
        difficulty,
        {"exit1": 0.7, "exit2": 0.8},
        lambda_value=0.2,
        minimum=0.55,
        maximum=0.99,
    )

    assert torch.allclose(thresholds["exit1"], torch.tensor([0.6, 0.8]))
    assert torch.allclose(thresholds["exit2"], torch.tensor([0.75, 0.85]))


def test_routing_uses_threshold_for_the_predicted_class() -> None:
    logits = {
        "exit1": torch.tensor([[4.0, 0.0], [0.0, 2.0], [0.2, 0.1]]),
        "exit2": torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.1, 0.2]]),
        "final": torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]),
    }
    thresholds = {
        "exit1": torch.tensor([0.90, 0.95]),
        "exit2": torch.tensor([0.99, 0.80]),
    }

    result = route_cached_logits(logits, {"exit1": 1.0, "exit2": 1.0, "final": 1.0}, thresholds)

    assert result.exit_indices.tolist() == [0, 1, 2]
    assert result.predictions.tolist() == [0, 1, 1]


def test_search_selects_lowest_flops_feasible_thresholds() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    logits = {
        "exit1": torch.tensor([[4.0, 0.0], [0.0, 0.8], [0.0, 4.0], [0.8, 0.0]]),
        "exit2": torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]),
        "final": torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]),
    }

    result = search_thresholds(
        logits=logits,
        labels=labels,
        temperatures={"exit1": 1.0, "exit2": 1.0, "final": 1.0},
        class_names=["zero", "one"],
        base_threshold_grid=[0.6, 0.8, 0.99],
        lambda_grid=[0.0],
        rho=20.0,
        exit_flops=[1.0, 2.0, 3.0],
        macro_f1_floor=0.99,
        recall_floors=[0.99, 0.99],
        split_name="threshold_validation",
    )

    assert result.best_row["tau_exit1"] == 0.8
    assert result.best_row["tau_exit2"] == 0.6
    assert math.isclose(result.best_row["average_flops"], 1.5)
    assert result.best_row["feasible"]


def test_threshold_search_rejects_locked_test_labels() -> None:
    logits = {name: torch.tensor([[1.0, 0.0], [0.0, 1.0]]) for name in ("exit1", "exit2", "final")}

    with pytest.raises(ValueError, match="threshold_validation"):
        search_thresholds(
            logits=logits,
            labels=torch.tensor([0, 1]),
            temperatures={"exit1": 1.0, "exit2": 1.0, "final": 1.0},
            class_names=["zero", "one"],
            base_threshold_grid=[0.8],
            lambda_grid=[0.0],
            rho=20.0,
            exit_flops=[1.0, 2.0, 3.0],
            macro_f1_floor=0.0,
            recall_floors=[0.0, 0.0],
            split_name="test",
        )

