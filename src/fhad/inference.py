from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .model import UNet


@dataclass
class PredictionResult:
    probability_map: np.ndarray
    binary_mask: np.ndarray
    mean_probability: float
    foreground_ratio: float
    risk_label: str


def risk_from_foreground_ratio(ratio: float) -> str:
    """Heuristic risk estimate from segmented region size.

    NOTE: This is NOT a clinical diagnosis.
    """
    if ratio < 0.08 or ratio > 0.45:
        return "high_risk_pattern"
    if ratio < 0.12 or ratio > 0.38:
        return "moderate_risk_pattern"
    return "likely_normal_pattern"


class FetalHeadInferenceService:
    def __init__(self, checkpoint: Path, image_size: int = 256, threshold: float = 0.5):
        self.checkpoint = Path(checkpoint)
        self.image_size = image_size
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet().to(self.device)
        self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
        self.model.eval()

    def _prepare_image(self, path: Path) -> tuple[np.ndarray, tuple[int, int]]:
        with Image.open(path) as img:
            img = img.convert("L")
            original_size = img.size
            resized = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            arr = np.asarray(resized, dtype=np.float32) / 255.0
        return arr, original_size

    def predict_path(self, path: Path) -> PredictionResult:
        arr, _ = self._prepare_image(path)
        tensor = torch.from_numpy(arr[None, None, ...]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        binary = (probs >= self.threshold).astype(np.uint8) * 255
        mean_probability = float(probs.mean())
        foreground_ratio = float((binary > 0).mean())
        return PredictionResult(
            probability_map=probs,
            binary_mask=binary,
            mean_probability=mean_probability,
            foreground_ratio=foreground_ratio,
            risk_label=risk_from_foreground_ratio(foreground_ratio),
        )
