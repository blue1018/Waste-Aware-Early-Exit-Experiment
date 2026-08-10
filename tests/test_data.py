from pathlib import Path

import pandas as pd
import torch

from waste_early_exit.config import load_config
from waste_early_exit.data import (
    assert_no_split_leakage,
    assign_group_stratified_splits,
    build_dataloaders,
    build_manifest,
    select_mode_subset,
)
from waste_early_exit.eda import summarize_dataset


def row_for(manifest: pd.DataFrame, path: Path) -> pd.Series:
    return manifest.loc[manifest["path"] == str(path.resolve())].iloc[0]


def test_manifest_flags_corruption_and_groups_duplicates(small_image_dataset: dict[str, object]) -> None:
    root = small_image_dataset["root"]
    paths = small_image_dataset["paths"]

    manifest = build_manifest(root, phash_distance=4)

    assert bool(row_for(manifest, paths["corrupt"])["corrupt"])
    assert row_for(manifest, paths["exact_source"])["sha256"] == row_for(manifest, paths["exact_copy"])["sha256"]
    assert row_for(manifest, paths["exact_source"])["duplicate_group"] == row_for(manifest, paths["exact_copy"])["duplicate_group"]
    assert row_for(manifest, paths["near_source"])["duplicate_group"] == row_for(manifest, paths["near_copy"])["duplicate_group"]
    assert manifest.loc[~manifest["corrupt"], "class_name"].nunique() == 12


def test_group_stratified_split_is_deterministic_and_leak_free(small_image_dataset: dict[str, object]) -> None:
    manifest = build_manifest(small_image_dataset["root"], phash_distance=4)

    first = assign_group_stratified_splits(manifest, seed=42)
    second = assign_group_stratified_splits(manifest, seed=42)

    assert first[["relative_path", "split"]].equals(second[["relative_path", "split"]])
    assert set(first.loc[~first["corrupt"], "split"]) == {
        "train",
        "validation",
        "calibration",
        "threshold_validation",
        "test",
    }
    assert_no_split_leakage(first)
    group_split_counts = first.loc[~first["corrupt"]].groupby("duplicate_group")["split"].nunique()
    assert int(group_split_counts.max()) == 1


def test_smoke_subset_caps_each_class_and_split(tmp_path: Path, small_image_dataset: dict[str, object]) -> None:
    manifest = build_manifest(small_image_dataset["root"], phash_distance=4)
    split_df = assign_group_stratified_splits(manifest, seed=42)
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml",
        project_root=tmp_path,
        overrides={"data.smoke_train_per_class": 2, "data.smoke_eval_per_class": 1},
    )

    subset = select_mode_subset(split_df, config)
    counts = subset.groupby(["split", "class_name"]).size()

    assert int(counts.loc["train"].max()) <= 2
    for split in ("validation", "calibration", "threshold_validation", "test"):
        assert int(counts.loc[split].max()) <= 1


def test_dataloaders_return_expected_tensor_shape(tmp_path: Path, small_image_dataset: dict[str, object]) -> None:
    manifest = build_manifest(small_image_dataset["root"], phash_distance=4)
    split_df = assign_group_stratified_splits(manifest, seed=42)
    config = load_config(
        Path(__file__).parents[1] / "configs" / "smoke.yaml",
        project_root=tmp_path,
        overrides={
            "paths.dataset_root": str(small_image_dataset["root"]),
            "data.image_size": 32,
            "data.smoke_train_per_class": 2,
            "data.smoke_eval_per_class": 1,
            "training.batch_size": 4,
        },
    )
    subset = select_mode_subset(split_df, config)

    loaders = build_dataloaders(subset, config)
    images, labels, sample_ids = next(iter(loaders["train"]))

    assert images.ndim == 4
    assert tuple(images.shape[1:]) == (3, 32, 32)
    assert labels.dtype == torch.int64
    assert len(sample_ids) == images.shape[0]
    assert set(loaders) == {"train", "validation", "calibration", "threshold_validation", "test"}


def test_dataset_summary_matches_manifest(small_image_dataset: dict[str, object]) -> None:
    manifest = build_manifest(small_image_dataset["root"], phash_distance=4)

    summary = summarize_dataset(manifest)

    assert summary["overview"].loc[0, "valid_images"] == int((~manifest["corrupt"]).sum())
    assert summary["classes"]["count"].sum() == int((~manifest["corrupt"]).sum())
    assert set(summary) >= {"overview", "classes", "sizes", "duplicates"}
