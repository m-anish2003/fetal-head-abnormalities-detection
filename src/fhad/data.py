from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path


def patient_id_from_name(name: str) -> str:
    stem = Path(name).stem
    return stem.split("_")[0] if "_" in stem else stem


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def discover_pairs(images_dir: Path, masks_dir: Path) -> list[Pair]:
    images = {p.name: p for p in images_dir.iterdir() if _is_image(p)}
    masks = {p.name: p for p in masks_dir.iterdir() if _is_image(p)}

    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        message = ["Dataset mismatch detected."]
        if missing_masks:
            message.append(f"Missing masks: {missing_masks[:10]}")
        if missing_images:
            message.append(f"Missing images: {missing_images[:10]}")
        raise ValueError(" ".join(message))

    return [Pair(images[name], masks[name]) for name in sorted(images)]


def train_val_split(
    pairs: list[Pair],
    val_split: float,
    seed: int,
    split_by_patient: bool = True,
) -> tuple[list[Pair], list[Pair]]:
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    if not split_by_patient:
        items = pairs.copy()
        random.Random(seed).shuffle(items)
        n_val = max(1, int(len(items) * val_split))
        return items[n_val:], items[:n_val]

    grouped: dict[str, list[Pair]] = {}
    for p in pairs:
        pid = patient_id_from_name(p.image.name)
        grouped.setdefault(pid, []).append(p)

    patient_ids = list(grouped.keys())
    random.Random(seed).shuffle(patient_ids)
    n_val_groups = max(1, int(len(patient_ids) * val_split))
    val_ids = set(patient_ids[:n_val_groups])

    train, val = [], []
    for pid, group in grouped.items():
        (val if pid in val_ids else train).extend(group)
    return train, val


def _load_grayscale(path: Path, size: int, is_mask: bool = False) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
        resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
        img = img.resize((size, size), resample)
        return np.asarray(img, dtype=np.float32) / 255.0


def _simple_aug(image: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

    if rng.random() < 0.5:
        image = np.flipud(image).copy()
        mask = np.flipud(mask).copy()

    if rng.random() < 0.3:
        angle = rng.uniform(-8.0, 8.0)
        image_pil = Image.fromarray((image * 255).astype(np.uint8), mode="L").rotate(
            angle, resample=Image.Resampling.BILINEAR
        )
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L").rotate(
            angle, resample=Image.Resampling.NEAREST
        )
        image = np.asarray(image_pil, dtype=np.float32) / 255.0
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0

    if rng.random() < 0.5:
        gamma = rng.uniform(0.8, 1.2)
        image = np.clip(image ** gamma, 0.0, 1.0)

    if rng.random() < 0.5:
        contrast = rng.uniform(0.85, 1.15)
        image = np.clip((image - 0.5) * contrast + 0.5, 0.0, 1.0)

    return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, pairs: Iterable[Pair], image_size: int, augment: bool = False, seed: int = 42):
        self.pairs = list(pairs)
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.pairs[index]
        image = _load_grayscale(pair.image, self.image_size, is_mask=False)
        mask = _load_grayscale(pair.mask, self.image_size, is_mask=True)
        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            image, mask = _simple_aug(image, mask, self.rng)

        return torch.from_numpy(image[None, ...]), torch.from_numpy(mask[None, ...])
