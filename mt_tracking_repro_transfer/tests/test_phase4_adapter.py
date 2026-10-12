"""Phase 4 CTDA adapter, tracker integration and training smoke tests."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.ctda.adapter import CTDAAssociation
from src.models.ctda.model import CTDANet, CTDAConfig
from src.simulation.dataset import Detection
from src.simulation.scenarios import generate_scenario, resolve_scenarios
from src.training.train_ctda import train
from src.tracking.baseline_tracker import GatedGreedyAssociation, MultiTargetTracker

CONFIG_PATH = Path("configs/phase4_ctda.yaml")


def _tiny_adapter(threshold: float = 0.1) -> CTDAAssociation:
    torch.manual_seed(0)
    config = CTDAConfig(variant="row_column_3ch", pad_size=8, channels=3, d_model=8, n_heads=2, dropout=0.0, enc_layers=1, dec_layers=1, ffn_dim=16)
    return CTDAAssociation(CTDANet(config), pad_size=8, threshold=threshold, one_to_one=True)


def _scenario_simulation(scenario="S0", distribution="baseline_distribution"):
    _, specs = resolve_scenarios(Path("configs/phase2_scenarios.yaml"), scenarios=[scenario], distributions=[distribution])
    return generate_scenario(specs[0])


class AdapterTests(unittest.TestCase):
    def test_adapter_returns_valid_one_to_one_pairs(self):
        adapter = _tiny_adapter()
        predicted = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 5.0]])
        measurements = np.array([[0.2, 0.1], [10.1, 9.9], [40.0, 40.0]])
        pairs = adapter.associate(predicted, measurements, gate=9.0)
        rows = [row for row, _ in pairs]
        columns = [column for _, column in pairs]
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(len(columns), len(set(columns)))
        for row, column in pairs:
            self.assertLess(row, len(predicted))
            self.assertLess(column, len(measurements))

    def test_adapter_is_deterministic(self):
        adapter = _tiny_adapter()
        predicted = np.array([[1.0, 2.0], [5.0, 5.0]])
        measurements = np.array([[1.1, 2.1], [30.0, 30.0]])
        self.assertEqual(adapter.associate(predicted, measurements, 9.0), adapter.associate(predicted, measurements, 9.0))

    def test_adapter_handles_empty_inputs(self):
        adapter = _tiny_adapter()
        self.assertEqual(adapter.associate(np.zeros((0, 2)), np.zeros((3, 2)), 9.0), [])
        self.assertEqual(adapter.associate(np.zeros((3, 2)), np.zeros((0, 2)), 9.0), [])

    def test_integrates_with_tracker_and_ignores_labels(self):
        simulation = _scenario_simulation("S0")
        adapter = _tiny_adapter()
        tracker = MultiTargetTracker(adapter, gate=9.0, min_hits=2, max_missed=5)
        labelled = [[Detection(d.position.copy(), d.source_target_id) for d in frame] for frame in simulation.detections]
        unlabelled = [[Detection(d.position.copy()) for d in frame] for frame in simulation.detections]
        first = tracker.run(labelled)
        tracker2 = MultiTargetTracker(_tiny_adapter(), gate=9.0, min_hits=2, max_missed=5)
        second = tracker2.run(unlabelled)
        for frame_a, frame_b in zip(first.estimates_by_frame, second.estimates_by_frame):
            self.assertEqual(len(frame_a), len(frame_b))
            for estimate_a, estimate_b in zip(frame_a, frame_b):
                self.assertEqual(estimate_a.track_id, estimate_b.track_id)
                self.assertTrue(np.array_equal(estimate_a.state, estimate_b.state))

    def test_adapter_matches_baseline_interface_contract(self):
        adapter = _tiny_adapter()
        self.assertTrue(callable(getattr(adapter, "associate", None)))
        baseline = GatedGreedyAssociation()
        predicted = np.array([[0.0, 0.0]])
        measurements = np.array([[0.1, 0.1]])
        for model in (adapter, baseline):
            pairs = model.associate(predicted, measurements, 9.0)
            self.assertIsInstance(pairs, list)
            for pair in pairs:
                self.assertEqual(len(pair), 2)


class TrainingSmokeTests(unittest.TestCase):
    def test_training_writes_checkpoint_and_summary(self):
        configuration = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        configuration["training"]["device"] = "cpu"
        with tempfile.TemporaryDirectory() as temporary_directory:
            summary = train(
                configuration,
                "row_column",
                Path(temporary_directory) / "row_column",
                {"epochs": 1, "samples": 4, "frames": 2},
            )
            self.assertEqual(summary["variant"], "row_column")
            self.assertEqual(summary["epochs"], 1)
            self.assertTrue(math.isfinite(summary["best_validation_loss"]))
            self.assertTrue(Path(summary["train_split"]).is_file())
            self.assertEqual(len(summary["train_split_sha256"]), 64)
            checkpoint = Path(temporary_directory) / "row_column" / "checkpoint.pt"
            self.assertTrue(checkpoint.is_file())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["variant"], "row_column")
            self.assertIn("state_dict", payload)


if __name__ == "__main__":
    unittest.main()
