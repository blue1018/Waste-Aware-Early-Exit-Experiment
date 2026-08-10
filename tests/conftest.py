from pathlib import Path

import numpy as np
import pytest
from PIL import Image


CLASS_NAMES = [
    "battery",
    "biological",
    "brown-glass",
    "cardboard",
    "clothes",
    "green-glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
    "white-glass",
]


@pytest.fixture(scope="session")
def small_image_dataset(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("images")
    rng = np.random.default_rng(2026)
    paths: dict[str, Path] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        class_dir.mkdir()
        for image_index in range(12):
            height = 28 + image_index % 3
            width = 30 + image_index % 4
            pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            channel = pixels[:, :, class_index % 3].astype(np.int16)
            pixels[:, :, class_index % 3] = ((channel // 2 + class_index * 13) % 256).astype(np.uint8)
            path = class_dir / f"{class_name}_{image_index:02d}.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            paths[f"{class_name}_{image_index}"] = path

    exact_source = paths["battery_0"]
    exact_copy = root / "battery" / "battery_exact_copy.png"
    exact_copy.write_bytes(exact_source.read_bytes())
    paths["exact_source"] = exact_source
    paths["exact_copy"] = exact_copy

    near_source = paths["battery_1"]
    near_pixels = np.asarray(Image.open(near_source).convert("RGB")).copy()
    near_pixels[0, 0, 0] = (int(near_pixels[0, 0, 0]) + 1) % 256
    near_copy = root / "battery" / "battery_near_copy.png"
    Image.fromarray(near_pixels, mode="RGB").save(near_copy)
    paths["near_source"] = near_source
    paths["near_copy"] = near_copy

    corrupt = root / "trash" / "broken.jpg"
    corrupt.write_bytes(b"not-an-image")
    paths["corrupt"] = corrupt
    return {"root": root, "paths": paths, "classes": CLASS_NAMES}
