import tempfile
import unittest
from pathlib import Path


class TestDataUtils(unittest.TestCase):
    def test_discover_pairs_mismatch_raises(self):
        try:
            from fhad.data import discover_pairs
        except Exception as exc:  # pragma: no cover - env dependency guard
            self.skipTest(f"Dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            images = root / "images"
            masks = root / "masks"
            images.mkdir()
            masks.mkdir()

            (images / "1_HC.png").write_bytes(b"fake")
            with self.assertRaises(ValueError):
                discover_pairs(images, masks)


if __name__ == "__main__":
    unittest.main()
