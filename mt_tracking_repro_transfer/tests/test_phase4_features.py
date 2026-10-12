"""Phase 4 feature/tensor and dataset-split tests."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.ctda.features import (
    PackedAssociationDataset,
    build_feature_tensor,
    rdf_channel,
    split_sample_indices,
)


def _write_packed(path: Path, frames_per_sample, pad_unused=False):
    """Write a tiny packed NPZ.

    ``frames_per_sample`` is a list (per sample) of lists of (position, truth_id)
    where truth_id = -1 means clutter.
    """

    measurements = []
    ids = []
    offsets = []
    cursor = 0
    for sample in frames_per_sample:
        row = [cursor]
        for frame in sample:
            for position, truth_id in frame:
                measurements.append(position)
                ids.append(truth_id)
                cursor += 1
            row.append(cursor)
        offsets.append(row)
    np.savez(
        path,
        measurements=np.asarray(measurements, dtype=np.float64).reshape((-1, 2)),
        measurement_truth_ids=np.asarray(ids, dtype=np.int64),
        frame_offsets=np.asarray(offsets, dtype=np.int64),
    )


class RDFTests(unittest.TestCase):
    def test_rdf_channel_range_and_sign(self):
        d1 = np.array([[0.0, 0.0], [10.0, 10.0]])
        d2 = np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0]])
        channel = rdf_channel(d1, d2, axis=0, x_bias=1e-3, signed=True)
        self.assertEqual(channel.shape, (2, 3))
        self.assertTrue(np.all(channel >= -1.0) and np.all(channel <= 1.0))
        self.assertTrue(np.all(np.isfinite(channel)))
        # sign follows the direction of d2 - d1
        self.assertLessEqual(channel[0, 0], 0.0)
        unsigned = rdf_channel(d1, d2, axis=0, x_bias=1e-3, signed=False)
        self.assertTrue(np.all(unsigned <= 0.0))

    def test_feature_tensor_shape_and_padding(self):
        rows = np.array([[10.0, 10.0], [12.0, 9.0]])
        columns = np.array([[0.0, 0.0], [1.0, 1.0], [50.0, 50.0]])
        tensor = build_feature_tensor(rows, columns, 6, ("x_rdf", "y_rdf", "range_rdf"), 1e-3, -1.0)
        self.assertEqual(tensor.shape, (3, 6, 6))
        # padded region keeps the pad value
        self.assertTrue(np.allclose(tensor[:, 4:, :], -1.0))
        self.assertTrue(np.allclose(tensor[:, :, 4:], -1.0))
        self.assertTrue(np.all(np.isfinite(tensor)))

    def test_overflow_is_clipped(self):
        d1 = np.array([[0.0, 0.0]])
        d2 = np.array([[1e9, 0.0], [1e-12, 0.0]])
        channel = rdf_channel(d1, d2, axis=0, x_bias=1e-9)
        self.assertTrue(np.all(np.isfinite(channel)))


class DatasetTests(unittest.TestCase):
    def _dataset(self):
        frames = [
            [  # sample 0, frame 0
                ([0.0, 0.0], 0),
                ([10.0, 0.0], 1),
                ([50.0, 50.0], -1),
            ],
            [  # sample 0, frame 1: target 0 missed, target 1 present, clutter
                ([10.2, 0.1], 1),
                ([60.0, 60.0], -1),
            ],
            [  # sample 0, frame 2: both targets present
                ([0.5, 0.0], 0),
                ([10.4, 0.2], 1),
            ],
        ]
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "packed.npz"
        _write_packed(path, [frames])
        dataset = PackedAssociationDataset(path, pad_size=6, channels=("x_rdf", "y_rdf"))
        return temporary, dataset

    def test_rows_are_previous_true_detections_and_labels_use_ids(self):
        temporary, dataset = self._dataset()
        example = dataset.build_example(0, 1)
        self.assertEqual(int(example.row_mask.sum()), 2)   # two previous true detections
        self.assertEqual(int(example.col_mask.sum()), 2)   # one true + one clutter
        target = example.target
        self.assertEqual(example.target.shape, (6, 6))
        # clutter column has no positive
        self.assertEqual(float(target[:2, 1].sum()), 0.0)
        # matched target 1 -> positive at (row of id 1, col of id 1)
        self.assertEqual(float(target.sum()), 1.0)
        temporary.cleanup()

    def test_missed_and_unmatched_counts(self):
        temporary, dataset = self._dataset()
        example = dataset.build_example(0, 2)
        # Rows come from frame 1, where only target 1 was detected.
        rows = int(example.row_mask.sum())
        columns = int(example.col_mask.sum())
        positives = int(example.target.sum())
        self.assertEqual(rows, 1)
        self.assertEqual(columns, 2)
        self.assertEqual(positives, 1)
        temporary.cleanup()

    def test_labels_do_not_enter_features(self):
        """Same positions with different ids give identical features but different targets."""

        frames_a = [
            [([0.0, 0.0], 0), ([10.0, 0.0], 1)],
            [([0.5, 0.0], 0), ([10.5, 0.0], 1)],
        ]
        # Swap the two target ids in the second frame only, keeping identical positions.
        frames_b = [
            [([0.0, 0.0], 0), ([10.0, 0.0], 1)],
            [([0.5, 0.0], 1), ([10.5, 0.0], 0)],
        ]
        temporary = tempfile.TemporaryDirectory()
        path_a = Path(temporary.name) / "a.npz"
        path_b = Path(temporary.name) / "b.npz"
        _write_packed(path_a, [frames_a])
        _write_packed(path_b, [frames_b])
        dataset_a = PackedAssociationDataset(path_a, pad_size=4, channels=("x_rdf", "y_rdf"))
        dataset_b = PackedAssociationDataset(path_b, pad_size=4, channels=("x_rdf", "y_rdf"))
        example_a = dataset_a.build_example(0, 1)
        example_b = dataset_b.build_example(0, 1)
        # Features depend only on positions.
        self.assertTrue(np.array_equal(example_a.features, example_b.features))
        # Targets depend on the ids, so a swapped id layout flips the positive cells.
        self.assertFalse(np.array_equal(example_a.target, example_b.target))
        temporary.cleanup()

    def test_empty_frames_are_supported(self):
        frames = [
            [([0.0, 0.0], 0)],
            [],                              # no measurements at all
            [([1.0, 0.0], 0), ([2.0, 0.0], -1)],
        ]
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "packed.npz"
        _write_packed(path, [frames])
        dataset = PackedAssociationDataset(path, pad_size=4, channels=("x_rdf",))
        example = dataset.build_example(0, 1)
        self.assertEqual(int(example.col_mask.sum()), 0)
        self.assertTrue(np.allclose(example.features, -1.0))
        self.assertEqual(float(example.target.sum()), 0.0)
        temporary.cleanup()

    def test_sequence_level_split_is_disjoint_and_deterministic(self):
        training, validation = split_sample_indices(100, 0.1, 7)
        self.assertEqual(len(training) + len(validation), 100)
        self.assertEqual(set(training).intersection(validation), set())
        again = split_sample_indices(100, 0.1, 7)
        self.assertEqual((training, validation), again)


if __name__ == "__main__":
    unittest.main()
