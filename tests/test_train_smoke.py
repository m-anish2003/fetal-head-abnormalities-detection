import tempfile
import unittest
from pathlib import Path


class TestTrainingSmoke(unittest.TestCase):
    def test_one_epoch_smoke(self):
        try:
            import numpy as np
            from PIL import Image
            from fhad.config import TrainConfig
            from fhad.train import run_training
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            train_i = root / "train" / "images"
            train_m = root / "train" / "masks"
            test_i = root / "test" / "images"
            test_m = root / "test" / "masks"
            for p in [train_i, train_m, test_i, test_m]:
                p.mkdir(parents=True, exist_ok=True)

            for idx in range(4):
                arr = np.zeros((32, 32), dtype=np.uint8)
                arr[8:24, 8:24] = 200
                mask = np.zeros((32, 32), dtype=np.uint8)
                mask[10:22, 10:22] = 255
                Image.fromarray(arr).save(train_i / f"{idx}_HC.png")
                Image.fromarray(mask).save(train_m / f"{idx}_HC.png")

            Image.fromarray(arr).save(test_i / "9_HC.png")
            Image.fromarray(mask).save(test_m / "9_HC.png")

            cfg = TrainConfig(
                train_images_dir=train_i,
                train_masks_dir=train_m,
                test_images_dir=test_i,
                test_masks_dir=test_m,
                output_dir=root / "artifacts",
                epochs=1,
                batch_size=2,
                image_size=32,
                num_workers=0,
                mixed_precision=False,
                early_stopping_patience=2,
            )
            summary = run_training(cfg)
            self.assertIn("best_val_dice", summary)
            self.assertTrue((root / "artifacts" / "best_model.pt").exists())


if __name__ == "__main__":
    unittest.main()
