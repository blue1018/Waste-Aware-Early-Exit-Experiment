from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import ExperimentConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_NAMES = ("train", "validation", "calibration", "threshold_validation", "test")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass
class _BKNode:
    value: int
    index: int
    children: dict[int, "_BKNode"]


class _BKTree:
    def __init__(self) -> None:
        self.root: _BKNode | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, index, {})
            return
        node = self.root
        while True:
            distance = self.distance(value, node.value)
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, index, {})
                return
            node = child

    def search(self, value: int, max_distance: int) -> list[int]:
        if self.root is None:
            return []
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self.distance(value, node.value)
            if distance <= max_distance:
                matches.append(node.index)
            low = distance - max_distance
            high = distance + max_distance
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)
        return matches


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(image: Image.Image) -> int:
    resized = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.int16)
    differences = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in differences.flatten():
        value = (value << 1) | int(bit)
    return value


def _image_record(path: Path, dataset_root: Path, class_index: int) -> dict[str, Any]:
    base = {
        "path": str(path.resolve()),
        "relative_path": path.relative_to(dataset_root).as_posix(),
        "class_name": path.parent.name,
        "class_index": class_index,
        "size_bytes": path.stat().st_size,
        "width": np.nan,
        "height": np.nan,
        "aspect_ratio": np.nan,
        "mode": "",
        "sha256": "",
        "phash": "",
        "corrupt": False,
        "error": "",
    }
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            base.update(
                {
                    "width": int(width),
                    "height": int(height),
                    "aspect_ratio": float(width / height),
                    "mode": image.mode,
                    "sha256": _sha256(path),
                    "phash": f"{_perceptual_hash(rgb):016x}",
                }
            )
    except Exception as error:
        base["corrupt"] = True
        base["error"] = f"{type(error).__name__}: {error}"
    return base


def _assign_duplicate_groups(manifest: pd.DataFrame, phash_distance: int) -> pd.DataFrame:
    result = manifest.copy()
    valid_indices = result.index[~result["corrupt"]].tolist()
    union_find = _UnionFind(len(result))
    sha_first: dict[str, int] = {}
    tree = _BKTree()
    for index in valid_indices:
        sha = str(result.at[index, "sha256"])
        if sha in sha_first:
            union_find.union(index, sha_first[sha])
        else:
            sha_first[sha] = index
        hash_value = int(str(result.at[index, "phash"]), 16)
        for neighbor in tree.search(hash_value, phash_distance):
            union_find.union(index, neighbor)
        tree.add(hash_value, index)

    root_to_group: dict[int, str] = {}
    groups: list[str] = []
    for index, row in result.iterrows():
        if bool(row["corrupt"]):
            groups.append(f"corrupt-{index:06d}")
            continue
        root = union_find.find(index)
        if root not in root_to_group:
            root_to_group[root] = f"group-{len(root_to_group):06d}"
        groups.append(root_to_group[root])
    result["duplicate_group"] = groups
    class_counts = result.loc[~result["corrupt"]].groupby("duplicate_group")["class_name"].nunique()
    conflict_groups = set(class_counts[class_counts > 1].index)
    result["duplicate_class_conflict"] = result["duplicate_group"].isin(conflict_groups)
    return result


def build_manifest(
    dataset_root: str | Path,
    output_path: str | Path | None = None,
    phash_distance: int = 4,
) -> pd.DataFrame:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {root}")
    class_to_index = {path.name: index for index, path in enumerate(class_dirs)}
    image_paths = sorted(
        path
        for class_dir in class_dirs
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    records = [_image_record(path, root, class_to_index[path.parent.name]) for path in image_paths]
    manifest = pd.DataFrame.from_records(records).sort_values("relative_path").reset_index(drop=True)
    manifest = _assign_duplicate_groups(manifest, phash_distance)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(destination, index=False)
    return manifest


def assign_group_stratified_splits(manifest: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    required = {"class_name", "duplicate_group", "corrupt", "relative_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    result = manifest.copy()
    result["fold"] = -1
    result["split"] = "excluded"
    valid = result.loc[~result["corrupt"]].copy()
    if valid["class_name"].nunique() < 2:
        raise ValueError("At least two classes are required for a stratified split")
    group_counts = valid.groupby("class_name")["duplicate_group"].nunique()
    if int(group_counts.min()) < 10:
        raise ValueError("Each class needs at least 10 duplicate groups for the 70/10/10/10 split")
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (_, fold_indices) in enumerate(
        splitter.split(valid, y=valid["class_name"], groups=valid["duplicate_group"])
    ):
        result.loc[valid.index[fold_indices], "fold"] = fold
    fold_to_split = {
        0: "train",
        1: "train",
        2: "train",
        3: "train",
        4: "train",
        5: "train",
        6: "validation",
        7: "calibration",
        8: "threshold_validation",
        9: "test",
    }
    valid_mask = ~result["corrupt"]
    result.loc[valid_mask, "split"] = result.loc[valid_mask, "fold"].map(fold_to_split)
    assert_no_split_leakage(result)
    return result


def assert_no_split_leakage(split_df: pd.DataFrame) -> None:
    valid = split_df.loc[~split_df["corrupt"]]
    if valid["relative_path"].duplicated().any():
        raise AssertionError("A sample appears more than once")
    if valid["split"].isna().any() or (valid["split"] == "excluded").any():
        raise AssertionError("A valid sample has no assigned split")
    group_split_counts = valid.groupby("duplicate_group")["split"].nunique()
    leaking = group_split_counts[group_split_counts > 1]
    if not leaking.empty:
        raise AssertionError(f"Duplicate groups cross splits: {list(leaking.index[:5])}")


def select_mode_subset(split_df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    valid = split_df.loc[~split_df["corrupt"]].copy()
    if config.mode == "full":
        return valid.sort_values("relative_path").reset_index(drop=True)
    selected: list[pd.DataFrame] = []
    for (split, _class_name), group in valid.groupby(["split", "class_name"], sort=True):
        cap = (
            config.data.smoke_train_per_class
            if split == "train"
            else config.data.smoke_eval_per_class
        )
        selected.append(group.sample(n=min(cap, len(group)), random_state=config.seed))
    return pd.concat(selected, ignore_index=True).sort_values(["split", "class_name", "relative_path"]).reset_index(drop=True)


class ManifestImageDataset(Dataset[tuple[torch.Tensor, int, str]]):
    def __init__(self, frame: pd.DataFrame, transform: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["class_index"]), str(row["relative_path"])


def _transforms(image_size: int) -> tuple[Any, Any]:
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 256 / 224))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(split_df: pd.DataFrame, config: ExperimentConfig) -> dict[str, DataLoader[Any]]:
    train_transform, eval_transform = _transforms(config.data.image_size)
    loaders: dict[str, DataLoader[Any]] = {}
    generator = torch.Generator().manual_seed(config.seed)
    for split in SPLIT_NAMES:
        frame = split_df.loc[split_df["split"] == split].copy()
        if frame.empty:
            raise ValueError(f"Split is empty: {split}")
        dataset = ManifestImageDataset(frame, train_transform if split == "train" else eval_transform)
        loaders[split] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=split == "train",
            num_workers=config.data.num_workers,
            pin_memory=False,
            drop_last=False,
            generator=generator,
            persistent_workers=config.data.num_workers > 0,
        )
    return loaders


def class_names_from_frame(frame: pd.DataFrame) -> list[str]:
    indexed = frame[["class_index", "class_name"]].drop_duplicates().sort_values("class_index")
    return indexed["class_name"].tolist()
