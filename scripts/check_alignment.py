from __future__ import annotations

import argparse
from pathlib import Path

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_names(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check image/mask alignment by filename")
    parser.add_argument("--images", type=Path, default=Path("data/train/images"))
    parser.add_argument("--masks", type=Path, default=Path("data/train/masks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_filenames = image_names(args.images)
    mask_filenames = image_names(args.masks)

    missing_masks = sorted(image_filenames - mask_filenames)
    missing_images = sorted(mask_filenames - image_filenames)

    print(f"Total training images: {len(image_filenames)}")
    print(f"Total training masks: {len(mask_filenames)}")

    if missing_masks:
        print("Images missing masks (up to 10 shown):")
        for m in missing_masks[:10]:
            print(" -", m)
    else:
        print("All images have corresponding masks!")

    if missing_images:
        print("Masks missing images (up to 10 shown):")
        for m in missing_images[:10]:
            print(" -", m)
    else:
        print("No masks are orphaned.")


if __name__ == "__main__":
    main()
