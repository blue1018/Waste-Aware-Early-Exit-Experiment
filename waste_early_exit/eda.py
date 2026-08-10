from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image


def summarize_dataset(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = manifest.loc[~manifest["corrupt"]].copy()
    duplicate_sizes = valid.groupby("duplicate_group").size()
    overview = pd.DataFrame(
        [
            {
                "discovered_files": len(manifest),
                "valid_images": len(valid),
                "corrupt_images": int(manifest["corrupt"].sum()),
                "classes": valid["class_name"].nunique(),
                "duplicate_groups": int((duplicate_sizes > 1).sum()),
                "cross_class_conflicts": int(valid["duplicate_class_conflict"].sum()),
            }
        ]
    )
    classes = valid.groupby("class_name").size().rename("count").reset_index()
    classes["share"] = classes["count"] / classes["count"].sum()
    sizes = valid[["width", "height", "aspect_ratio", "size_bytes"]].describe().T.reset_index(names="measure")
    duplicates = (
        valid.groupby("duplicate_group")
        .agg(size=("relative_path", "size"), classes=("class_name", "nunique"))
        .query("size > 1")
        .sort_values(["size", "duplicate_group"], ascending=[False, True])
        .reset_index()
    )
    return {"overview": overview, "classes": classes, "sizes": sizes, "duplicates": duplicates}


def _save_figure(figure: plt.Figure, output_path: str | Path | None) -> plt.Figure:
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    return figure


def plot_class_distribution(manifest: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    counts = (
        manifest.loc[~manifest["corrupt"]]
        .groupby("class_name")
        .size()
        .sort_values(ascending=True)
        .rename("Images")
    )
    figure, axis = plt.subplots(figsize=(8, 5))
    counts.plot.barh(ax=axis, color="#3B82F6")
    axis.set_title("Class distribution")
    axis.set_xlabel("Images")
    axis.set_ylabel("Class")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_image_sizes(manifest: pd.DataFrame, output_path: str | Path | None = None) -> plt.Figure:
    valid = manifest.loc[~manifest["corrupt"]]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.scatterplot(data=valid, x="width", y="height", hue="class_name", legend=False, s=16, alpha=0.5, ax=axes[0])
    axes[0].set_title("Image dimensions")
    sns.histplot(data=valid, x="aspect_ratio", bins=30, color="#14B8A6", ax=axes[1])
    axes[1].set_title("Aspect ratio")
    figure.tight_layout()
    return _save_figure(figure, output_path)


def plot_sample_grid(
    manifest: pd.DataFrame,
    output_path: str | Path | None = None,
    seed: int = 42,
) -> plt.Figure:
    valid = manifest.loc[~manifest["corrupt"]]
    samples = valid.groupby("class_name", group_keys=False).sample(n=1, random_state=seed)
    columns = 4
    rows = int(np.ceil(len(samples) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10, 2.6 * rows))
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, (_, row) in zip(flat_axes, samples.iterrows()):
        with Image.open(row["path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(str(row["class_name"]))
        axis.axis("off")
    for axis in flat_axes[len(samples) :]:
        axis.axis("off")
    figure.suptitle("One sample per class", y=1.01)
    figure.tight_layout()
    return _save_figure(figure, output_path)


def split_count_table(split_df: pd.DataFrame) -> pd.DataFrame:
    return (
        split_df.loc[~split_df["corrupt"]]
        .pivot_table(index="class_name", columns="split", values="relative_path", aggfunc="count", fill_value=0)
        .reset_index()
    )

