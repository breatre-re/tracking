"""Phase 4 CTDA model, loss and decoding tests."""

import unittest

import numpy as np
import torch

from src.models.ctda.model import CTDANet, CTDAConfig
from src.models.ctda.objective import association_metrics, decode_association, focal_loss


def _tiny_config(variant: str, channels: int = 2, pad: int = 8) -> CTDAConfig:
    return CTDAConfig(
        variant=variant,
        pad_size=pad,
        channels=channels,
        d_model=pad,
        n_heads=2,
        dropout=0.0,
        enc_layers=1,
        dec_layers=1,
        ffn_dim=16,
    )


class ModelTests(unittest.TestCase):
    def test_all_variants_forward_shapes_and_finite(self):
        batch, pad, channels = 2, 8, 2
        features = torch.rand(batch, channels, pad, pad)
        row_mask = torch.ones(batch, pad, dtype=torch.bool)
        col_mask = torch.ones(batch, pad, dtype=torch.bool)
        for variant in ("standard_transformer", "row_column", "row_column_3ch"):
            model = CTDANet(_tiny_config(variant, channels))
            logits = model(features, row_mask, col_mask)
            self.assertEqual(logits.shape, (batch, pad, pad), variant)
            self.assertTrue(torch.isfinite(logits).all(), variant)

    def test_masked_cells_are_ignored(self):
        pad, channels = 8, 2
        features = torch.rand(1, channels, pad, pad)
        row_mask = torch.ones(1, pad, dtype=torch.bool)
        col_mask = torch.ones(1, pad, dtype=torch.bool)
        row_mask[0, 5:] = False
        col_mask[0, 6:] = False
        model = CTDANet(_tiny_config("row_column_3ch", channels))
        logits = model(features, row_mask, col_mask)
        self.assertTrue(torch.all(logits[0, 5:, :] <= -1e8))
        self.assertTrue(torch.all(logits[0, :, 6:] <= -1e8))
        # padded region must never carry a normal logit
        self.assertFalse(torch.any(logits[0, :5, :6] <= -1e8))

    def test_row_column_variants_require_square_width(self):
        with self.assertRaises(ValueError):
            CTDANet(CTDAConfig(variant="row_column", pad_size=8, channels=2, d_model=4))

    def test_padded_region_does_not_affect_valid_region(self):
        """Changing padded cells must not change valid-region logits (mask contract)."""

        pad, channels = 8, 2
        torch.manual_seed(0)
        model = CTDANet(_tiny_config("row_column_3ch", channels)).eval()
        features = torch.rand(1, channels, pad, pad)
        row_mask = torch.ones(1, pad, dtype=torch.bool)
        col_mask = torch.ones(1, pad, dtype=torch.bool)
        row_mask[0, 5:] = False
        col_mask[0, 6:] = False
        mutated = features.clone()
        mutated[:, :, 5:, :] = 123.0
        mutated[:, :, :, 6:] = -456.0
        with torch.no_grad():
            base = model(features, row_mask, col_mask)
            changed = model(mutated, row_mask, col_mask)
        self.assertTrue(torch.allclose(base[0, :5, :6], changed[0, :5, :6], atol=1e-5))


class ObjectiveTests(unittest.TestCase):
    def test_focal_loss_ignores_padding(self):
        pad = 6
        logits = torch.randn(1, pad, pad)
        targets = torch.zeros(1, pad, pad)
        targets[0, 0, 0] = 1.0
        row_mask = torch.ones(1, pad, dtype=torch.bool)
        col_mask = torch.ones(1, pad, dtype=torch.bool)
        row_mask[0, 4:] = False
        col_mask[0, 4:] = False
        base = focal_loss(logits, targets, row_mask, col_mask)
        mutated = logits.clone()
        mutated[0, 4:, :] = 999.0
        mutated[0, :, 4:] = -999.0
        self.assertAlmostEqual(float(base), float(focal_loss(mutated, targets, row_mask, col_mask)), places=6)

    def test_decode_is_one_to_one_and_thresholded(self):
        pad = 4
        probabilities = np.zeros((pad, pad), dtype=float)
        probabilities[0, 0] = 0.9
        probabilities[0, 1] = 0.8      # cannot also take row 0
        probabilities[1, 0] = 0.7      # cannot also take column 0
        probabilities[2, 2] = 0.05     # below threshold
        row_mask = np.ones(pad, dtype=bool)
        col_mask = np.ones(pad, dtype=bool)
        association = decode_association(probabilities, row_mask, col_mask, threshold=0.1, one_to_one=True)
        # Optimal assignment keeps the two high-scoring, non-conflicting pairs.
        self.assertEqual(int(association.sum()), 2)
        rows, columns = np.nonzero(association)
        self.assertEqual(len(set(rows.tolist())), len(rows))
        self.assertEqual(len(set(columns.tolist())), len(columns))
        # (0, 0) loses to the better global assignment and must stay unset.
        self.assertFalse(association[0, 0])

    def test_threshold_without_one_to_one(self):
        pad = 3
        probabilities = np.full((pad, pad), 0.5)
        association = decode_association(
            probabilities, np.ones(pad, dtype=bool), np.ones(pad, dtype=bool), threshold=0.1, one_to_one=False
        )
        self.assertEqual(int(association.sum()), pad * pad)

    def test_metrics_perfect_prediction(self):
        pad = 5
        target = np.zeros((pad, pad), dtype=np.float32)
        target[0, 1] = 1.0
        probabilities = np.zeros((pad, pad), dtype=float)
        probabilities[0, 1] = 0.99
        metrics = association_metrics(probabilities, target, np.ones(pad, dtype=bool), np.ones(pad, dtype=bool))
        self.assertEqual(metrics["true_positives"], 1.0)
        self.assertEqual(metrics["false_positives"], 0.0)
        self.assertEqual(metrics["false_negatives"], 0.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_empty_example_metrics(self):
        pad = 4
        target = np.zeros((pad, pad), dtype=np.float32)
        metrics = association_metrics(
            np.zeros((pad, pad)), target, np.zeros(pad, dtype=bool), np.zeros(pad, dtype=bool)
        )
        self.assertEqual(metrics["cells"], 0.0)
        self.assertEqual(metrics["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
