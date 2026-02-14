import unittest


class TestMetrics(unittest.TestCase):
    def test_dice_iou_sanity(self):
        try:
            import torch
            from fhad.metrics import dice_score, iou_score
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"Dependencies unavailable: {exc}")

        # perfect prediction -> score close to 1
        target = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        logits = torch.full((1, 1, 4, 4), 20.0)
        self.assertGreater(float(dice_score(logits, target)), 0.99)
        self.assertGreater(float(iou_score(logits, target)), 0.99)

        # opposite prediction -> score low
        bad_logits = torch.full((1, 1, 4, 4), -20.0)
        self.assertLess(float(dice_score(bad_logits, target)), 0.05)
        self.assertLess(float(iou_score(bad_logits, target)), 0.05)


if __name__ == "__main__":
    unittest.main()
