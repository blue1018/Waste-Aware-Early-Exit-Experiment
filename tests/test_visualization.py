from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")

from waste_early_exit.visualization import (
    plot_confusion_matrix,
    plot_exit_shares,
    plot_pareto,
    plot_qualitative_examples,
    plot_training_history,
)


def test_main_figures_are_returned_and_saved(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [1.0, 0.8],
            "validation_loss": [1.1, 0.9],
            "train_macro_f1": [0.3, 0.5],
            "validation_macro_f1": [0.2, 0.4],
        }
    )
    figures = [
        plot_training_history(history, tmp_path / "training.png"),
        plot_confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], ["zero", "one"], tmp_path / "confusion.png"),
        plot_exit_shares(
            pd.DataFrame({"method": ["global", "proposed"], "exit1": [0.2, 0.4], "exit2": [0.3, 0.2], "final": [0.5, 0.4]}),
            tmp_path / "exits.png",
        ),
        plot_pareto(
            pd.DataFrame({"method": ["a", "b"], "average_flops": [1.0, 2.0], "macro_f1": [0.7, 0.8]}),
            x="average_flops",
            output_path=tmp_path / "pareto.png",
        ),
    ]

    assert all(figure.axes for figure in figures)
    assert {path.name for path in tmp_path.glob("*.png")} == {
        "training.png",
        "confusion.png",
        "exits.png",
        "pareto.png",
    }


def test_qualitative_examples_use_sample_ids_and_exit_labels(tmp_path: Path) -> None:
    paths = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0))):
        path = tmp_path / f"sample_{index}.png"
        Image.new("RGB", (12, 12), color=color).save(path)
        paths.append(path)
    frame = pd.DataFrame(
        {
            "relative_path": ["a.png", "b.png"],
            "path": [str(path) for path in paths],
        }
    )

    figure = plot_qualitative_examples(
        frame,
        ["a.png", "b.png"],
        np.array([0, 1]),
        np.array([0, 0]),
        np.array([0, 2]),
        ["zero", "one"],
        tmp_path / "examples.png",
    )

    assert figure.axes
    assert (tmp_path / "examples.png").exists()
