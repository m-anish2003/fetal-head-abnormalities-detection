from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .data import VALID_EXTENSIONS
from .inference import FetalHeadInferenceService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict segmentation masks for single image or folder")
    p.add_argument("--input", type=Path, required=True, help="Input image path or folder")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("predictions"))
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def collect_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def main() -> None:
    args = parse_args()
    service = FetalHeadInferenceService(args.checkpoint, image_size=args.image_size, threshold=args.threshold)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = collect_inputs(args.input)
    summary = []

    for img_path in images:
        with Image.open(img_path) as img:
            original_size = img.convert("L").size

        pred = service.predict_path(img_path)

        mask_img = Image.fromarray(pred.binary_mask, mode="L").resize(original_size, Image.Resampling.NEAREST)
        out_png = args.output_dir / f"{img_path.stem}_mask.png"
        mask_img.save(out_png)

        item = {
            "input": str(img_path),
            "mask": str(out_png),
            "mean_probability": pred.mean_probability,
            "foreground_ratio": pred.foreground_ratio,
            "risk_label": pred.risk_label,
        }
        summary.append(item)

    out_json = args.output_dir / "predictions_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"outputs": len(summary), "summary_file": str(out_json)}, indent=2))


if __name__ == "__main__":
    main()
