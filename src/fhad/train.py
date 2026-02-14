from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import SegmentationDataset, discover_pairs, train_val_split
from .metrics import dice_loss, dice_score, iou_score
from .model import UNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SegmentationLoss(nn.Module):
    def __init__(self, mode: str = "bce_dice", dice_weight: float = 0.5):
        super().__init__()
        self.mode = mode
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.mode == "bce":
            return self.bce(logits, target)
        if self.mode == "dice":
            return dice_loss(logits, target)
        if self.mode == "bce_dice":
            return (1.0 - self.dice_weight) * self.bce(logits, target) + self.dice_weight * dice_loss(logits, target)
        raise ValueError(f"Unsupported loss mode: {self.mode}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module) -> dict[str, float]:
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_dice += dice_score(logits, y).item()
            total_iou += iou_score(logits, y).item()

    n = max(1, len(loader))
    return {"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n}


def _make_loader(ds: SegmentationDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def run_training(cfg: TrainConfig) -> dict[str, float]:
    set_seed(cfg.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_pairs = discover_pairs(cfg.train_images_dir, cfg.train_masks_dir)
    if cfg.val_images_dir and cfg.val_masks_dir:
        val_pairs = discover_pairs(cfg.val_images_dir, cfg.val_masks_dir)
    else:
        train_pairs, val_pairs = train_val_split(
            train_pairs, cfg.val_split, cfg.random_seed, split_by_patient=cfg.split_by_patient
        )

    test_pairs = (
        discover_pairs(cfg.test_images_dir, cfg.test_masks_dir)
        if cfg.test_images_dir and cfg.test_masks_dir
        else None
    )

    train_ds = SegmentationDataset(train_pairs, cfg.image_size, augment=cfg.use_augmentation, seed=cfg.random_seed)
    val_ds = SegmentationDataset(val_pairs, cfg.image_size, augment=False, seed=cfg.random_seed)
    test_ds = SegmentationDataset(test_pairs, cfg.image_size, augment=False, seed=cfg.random_seed) if test_pairs else None

    train_loader = _make_loader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = _make_loader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = _make_loader(test_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers) if test_ds else None

    model = UNet().to(device)
    criterion = SegmentationLoss(mode=cfg.loss, dice_weight=cfg.dice_weight)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=cfg.mixed_precision and device.type == "cuda")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_dice, patience = -1.0, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.mixed_precision and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device, criterion)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)
        print(json.dumps(row))

        if row["val_dice"] > best_dice:
            best_dice = row["val_dice"]
            patience = 0
            torch.save(model.state_dict(), cfg.output_dir / "best_model.pt")
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(cfg.output_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device, criterion) if test_loader else None

    summary = {
        "best_val_dice": best_dice,
        "epochs_ran": len(history),
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds) if test_ds else 0,
        "loss": cfg.loss,
        "augmentation": cfg.use_augmentation,
    }
    result = {"summary": summary, "history": history, "test_metrics": test_metrics}
    (cfg.output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    return summary


def run_benchmark(base_cfg: TrainConfig) -> None:
    experiments = [
        {"name": "baseline_bce_noaug", "loss": "bce", "use_augmentation": False},
        {"name": "bce_dice_aug", "loss": "bce_dice", "use_augmentation": True},
    ]
    results = []
    for exp in experiments:
        cfg = copy.deepcopy(base_cfg)
        cfg.loss = exp["loss"]
        cfg.use_augmentation = exp["use_augmentation"]
        cfg.output_dir = base_cfg.output_dir / exp["name"]
        summary = run_training(cfg)
        summary["name"] = exp["name"]
        results.append(summary)

    benchmark_path = base_cfg.output_dir / "benchmark.json"
    base_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(json.dumps(results, indent=2))
    print(f"Benchmark saved to {benchmark_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UNet for fetal head segmentation")
    p.add_argument("--train-images", type=Path, default=Path("data/train/images"))
    p.add_argument("--train-masks", type=Path, default=Path("data/train/masks"))
    p.add_argument("--val-images", type=Path, default=None)
    p.add_argument("--val-masks", type=Path, default=None)
    p.add_argument("--test-images", type=Path, default=None)
    p.add_argument("--test-masks", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--split-by-patient", action="store_true", default=True)
    p.add_argument("--no-split-by-patient", dest="split_by_patient", action="store_false")
    p.add_argument("--loss", choices=["bce", "dice", "bce_dice"], default="bce_dice")
    p.add_argument("--dice-weight", type=float, default=0.5)
    p.add_argument("--no-augmentation", action="store_true")
    p.add_argument("--benchmark", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        train_images_dir=args.train_images,
        train_masks_dir=args.train_masks,
        val_images_dir=args.val_images,
        val_masks_dir=args.val_masks,
        test_images_dir=args.test_images,
        test_masks_dir=args.test_masks,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        split_by_patient=args.split_by_patient,
        loss=args.loss,
        dice_weight=args.dice_weight,
        use_augmentation=not args.no_augmentation,
    )
    if args.benchmark:
        run_benchmark(cfg)
        return

    summary = run_training(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
