from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ANNOTATION_PATTERN = re.compile(r"_?annotation[s]?", flags=re.IGNORECASE)


def copy_split(raw_dir: Path, target_images: Path, target_masks: Path | None = None) -> None:
    target_images.mkdir(parents=True, exist_ok=True)
    if target_masks:
        target_masks.mkdir(parents=True, exist_ok=True)

    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue

        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue

        if target_masks and "annotation" in path.name.lower():
            clean_name = ANNOTATION_PATTERN.sub("", path.name)
            shutil.copy(path, target_masks / clean_name)
        else:
            shutil.copy(path, target_images / path.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Organize fetal head dataset into train/test folders")
    p.add_argument("--raw-train", type=Path, default=Path("raw_dataset/training_data"))
    p.add_argument("--raw-test", type=Path, default=Path("raw_dataset/test_data"))
    p.add_argument("--train-images", type=Path, default=Path("data/train/images"))
    p.add_argument("--train-masks", type=Path, default=Path("data/train/masks"))
    p.add_argument("--test-images", type=Path, default=Path("data/test/images"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    copy_split(args.raw_train, args.train_images, args.train_masks)
    print("Training data copied & renamed successfully!")

    copy_split(args.raw_test, args.test_images)
    print("Test data copied successfully!")


if __name__ == "__main__":
    main()
