from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SegmentationDataset, discover_pairs
from .metrics import dice_score, iou_score
from .model import UNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained UNet model")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pairs = discover_pairs(args.images, args.masks)
    ds = SegmentationDataset(pairs, args.image_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    total_loss = total_dice = total_iou = 0.0

    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_dice += dice_score(logits, y).item()
            total_iou += iou_score(logits, y).item()

    n = max(1, len(dl))
    print(json.dumps({"loss": total_loss / n, "dice": total_dice / n, "iou": total_iou / n, "samples": len(ds)}, indent=2))


if __name__ == "__main__":
    main()
