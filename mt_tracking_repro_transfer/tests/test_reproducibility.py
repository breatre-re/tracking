import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.simulation.dataset import (
    ScenarioConfig,
    load_dataset_sample,
    sample_with_events,
    write_dataset_split,
)


class DatasetReproducibilityTests(unittest.TestCase):
    def test_same_seed_produces_byte_identical_dataset(self):
        config = ScenarioConfig(steps=12, num_targets=4, seed=53)
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = Path(temporary_directory) / "first.npz"
            second = Path(temporary_directory) / "second.npz"
            for path in (first, second):
                write_dataset_split(
                    path,
                    split="smoke",
                    config=config,
                    sample_count=3,
                    seed_start=config.seed,
                )
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_every_measurement_has_truth_or_clutter_label(self):
        sample = sample_with_events(
            ScenarioConfig(steps=20, num_targets=4, detection_probability=0.55, seed=9)
        )
        for frame, (detections, truth_ids) in enumerate(
            zip(sample.simulation.detections, sample.measurement_truth_ids)
        ):
            self.assertEqual(len(detections), len(truth_ids))
            for detection, truth_id in zip(detections, truth_ids):
                if truth_id == -1:
                    self.assertIsNone(detection.source_target_id)
                else:
                    self.assertEqual(detection.source_target_id, int(truth_id))
                    self.assertTrue(
                        np.isfinite(sample.simulation.true_states[frame, truth_id, 0])
                    )

    def test_loader_restores_labels_and_events(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "train.npz"
            write_dataset_split(
                path,
                split="train",
                config=ScenarioConfig(steps=14, num_targets=3),
                sample_count=2,
                seed_start=100,
            )
            sample = load_dataset_sample(path, sample_index=1)
            self.assertEqual(sample.simulation.config.seed, 101)
            self.assertEqual(sample.simulation.true_states.shape, (14, 3, 4))
            self.assertEqual(len(sample.measurement_truth_ids), 14)
            for frame, detections in enumerate(sample.simulation.detections):
                self.assertEqual(
                    len(detections), len(sample.measurement_truth_ids[frame])
                )
                self.assertTrue(all(d.source_target_id is None for d in detections))
            for frame, local_index in sample.clutter_events:
                self.assertEqual(
                    sample.measurement_truth_ids[int(frame)][int(local_index)], -1
                )
            expected_misses = set()
            for frame, truth_state in enumerate(sample.simulation.true_states):
                active = np.flatnonzero(np.isfinite(truth_state[:, 0]))
                detected = set(
                    int(value)
                    for value in sample.measurement_truth_ids[frame]
                    if value >= 0
                )
                expected_misses.update(
                    (frame, int(target_id))
                    for target_id in active
                    if int(target_id) not in detected
                )
            self.assertEqual(
                expected_misses,
                {tuple(event) for event in sample.missed_detection_events.tolist()},
            )

    def test_distinct_split_seed_ranges_produce_distinct_samples(self):
        config = ScenarioConfig(steps=12, num_targets=3)
        with tempfile.TemporaryDirectory() as temporary_directory:
            train_path = Path(temporary_directory) / "train.npz"
            test_path = Path(temporary_directory) / "test.npz"
            write_dataset_split(
                train_path, split="train", config=config, sample_count=3, seed_start=20
            )
            write_dataset_split(
                test_path, split="test", config=config, sample_count=2, seed_start=30
            )
            train = [load_dataset_sample(train_path, i) for i in range(3)]
            test = [load_dataset_sample(test_path, i) for i in range(2)]
            train_hashes = {
                sample.simulation.true_states.tobytes() for sample in train
            }
            test_hashes = {
                sample.simulation.true_states.tobytes() for sample in test
            }
            self.assertTrue(train_hashes.isdisjoint(test_hashes))

    def test_variable_target_count_sampling_and_loader(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "paper_assumption_v1.npz"
            repeated_path = Path(temporary_directory) / "paper_assumption_v1_repeat.npz"
            config = ScenarioConfig(
                steps=12, num_targets=25, clutter_rate=20, seed=1
            )
            write_dataset_split(
                path,
                split="train",
                config=config,
                sample_count=50,
                seed_start=3000,
                target_count_range=(1, 25),
            )
            write_dataset_split(
                repeated_path,
                split="train",
                config=config,
                sample_count=50,
                seed_start=3000,
                target_count_range=(1, 25),
            )
            self.assertEqual(path.read_bytes(), repeated_path.read_bytes())
            with np.load(path, allow_pickle=False) as dataset:
                target_counts = dataset["target_counts"]
                self.assertGreaterEqual(int(target_counts.min()), 1)
                self.assertLessEqual(int(target_counts.max()), 25)
                self.assertGreater(len(np.unique(target_counts)), 1)
                self.assertEqual(dataset["truth_states"].shape, (50, 12, 25, 4))
            for index in (0, 24, 49):
                sample = load_dataset_sample(path, index)
                self.assertEqual(
                    sample.simulation.true_states.shape[1],
                    int(target_counts[index]),
                )
                self.assertTrue(
                    all(
                        detection.source_target_id is None
                        for frame in sample.simulation.detections
                        for detection in frame
                    )
                )


if __name__ == "__main__":
    unittest.main()
