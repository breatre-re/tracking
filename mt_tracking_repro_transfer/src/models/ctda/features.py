"""Phase 4 CTDA feature construction and association dataset.

Paper source: Section 3.1-3.2 (Eq. 6-10), p. 4. Candidate tracks (rows) and
measurements (columns) are described by RDF ("relative distance feature")
matrices, one per spatial axis; the matrix is then zero-padded to a standard
size for the cross Transformer.

Label-safe construction (no truth leakage into features):

* Rows ("candidate tracks") are built from the *previous frame's detections*
  (a synthesized candidate-history representation, since the packed data does
  not store trajectory histories). Rows are therefore detection positions, not
  truth states.
* Columns are the current frame's measurements (detections and clutter).
* The (i, j) label is 1 iff the previous detection and the current measurement
  carry the same source target id. Source ids are used **only** to build the
  supervision matrix; they never enter the feature tensor.
* A row with no positive column is an unmatched candidate (missed detection); a
  column with no positive row is clutter / a new target.

Reproduction assumptions are documented in ``configs/phase4_ctda.yaml`` and in
``docs/reports/reproduction_assumptions.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PAD_TARGET = 0.0


def rdf_channel(d1: np.ndarray, d2: np.ndarray, axis: int, x_bias: float, signed: bool = True) -> np.ndarray:
    """Relative distance feature for one axis (paper Eq. 6-7).

    Implemented as printed: ``c_ij = (2 / (1 + exp((|d| + bias) / mid)) - 1) * sign(d)``
    with ``mid = max(|d|) * (min(|d|) + bias)``. The extracted PDF also states
    the qualitative intent that elements near +/-1 mean "similar"; that sentence
    is inverted relative to the printed expression, so this is flagged as an
    unresolved ambiguity in the audit table (a learned model can invert any
    monotone feature).
    """

    d1_axis = np.asarray(d1, dtype=np.float64)[:, axis][:, None]
    d2_axis = np.asarray(d2, dtype=np.float64)[:, axis][None, :]
    diff = d2_axis - d1_axis
    absolute = np.abs(diff)
    if absolute.size == 0:
        return np.zeros((len(d1), len(d2)), dtype=np.float64)
    mid = float(absolute.max()) * (float(absolute.min()) + x_bias)
    if mid <= 0.0:
        mid = x_bias
    exponent = np.clip((absolute + x_bias) / mid, -60.0, 60.0)
    logistic = 2.0 / (1.0 + np.exp(exponent)) - 1.0
    if signed:
        logistic = logistic * np.sign(diff)
    return logistic


def build_feature_tensor(
    row_positions: np.ndarray,
    col_positions: np.ndarray,
    pad_size: int,
    channels: Tuple[str, ...],
    x_bias: float,
    pad_value: float,
) -> np.ndarray:
    """Return a ``(C, pad_size, pad_size)`` feature tensor for one example."""

    channels_out = np.full((len(channels), pad_size, pad_size), pad_value, dtype=np.float32)
    rows = int(min(len(row_positions), pad_size))
    columns = int(min(len(col_positions), pad_size))
    if rows == 0 or columns == 0:
        return channels_out
    row_positions = np.asarray(row_positions, dtype=np.float64)[:rows]
    col_positions = np.asarray(col_positions, dtype=np.float64)[:columns]
    for channel_index, name in enumerate(channels):
        if name == "x_rdf":
            block = rdf_channel(row_positions, col_positions, 0, x_bias, signed=True)
        elif name == "y_rdf":
            block = rdf_channel(row_positions, col_positions, 1, x_bias, signed=True)
        elif name == "range_rdf":
            # Sign-free magnitude channel (ASSUMPTION for the 2D 3-channel design).
            distance = np.linalg.norm(
                col_positions[None, :, :] - row_positions[:, None, :], axis=2
            )
            mid = float(distance.max()) * (float(distance.min()) + x_bias) if distance.size else x_bias
            mid = mid if mid > 0 else x_bias
            block = 2.0 / (1.0 + np.exp(np.clip((distance + x_bias) / mid, -60.0, 60.0))) - 1.0
        else:
            raise ValueError(f"unknown feature channel {name!r}")
        channels_out[channel_index, :rows, :columns] = block.astype(np.float32)
    return channels_out


@dataclass
class AssociationExample:
    features: np.ndarray   # (C, P, P)
    row_mask: np.ndarray   # (P,) bool
    col_mask: np.ndarray   # (P,) bool
    target: np.ndarray     # (P, P) float in {0,1}
    sample_index: int
    frame: int

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features),
            "row_mask": torch.from_numpy(self.row_mask),
            "col_mask": torch.from_numpy(self.col_mask),
            "target": torch.from_numpy(self.target),
        }


def _frame_slice(offsets: np.ndarray, sample_index: int, frame: int) -> Tuple[int, int]:
    return int(offsets[sample_index, frame]), int(offsets[sample_index, frame + 1])


class PackedAssociationDataset(Dataset):
    """Sequence-wise association examples derived from a packed Phase 1.x NPZ."""

    def __init__(
        self,
        path: Path,
        pad_size: int = 40,
        channels: Tuple[str, ...] = ("x_rdf", "y_rdf", "range_rdf"),
        x_bias: float = 0.001,
        pad_value: float = -1.0,
        sample_indices: Optional[List[int]] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as dataset:
            self.measurements = np.asarray(dataset["measurements"], dtype=np.float64)
            self.truth_ids = np.asarray(dataset["measurement_truth_ids"], dtype=np.int64)
            self.offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
        self.sample_count, self.steps_plus_one = self.offsets.shape
        self.steps = self.steps_plus_one - 1
        self.pad_size = pad_size
        self.channels = tuple(channels)
        self.x_bias = x_bias
        self.pad_value = pad_value
        if sample_indices is None:
            sample_indices = list(range(self.sample_count))
        self.sample_indices = list(sample_indices)
        self.max_frames = max_frames if max_frames is not None else self.steps
        # Index of (sample_index, frame) pairs; frame >= 1 because a row needs a
        # previous frame. Sequence-level split means no frame leakage between
        # train and test.
        self.index: List[Tuple[int, int]] = [
            (sample, frame)
            for sample in self.sample_indices
            for frame in range(1, min(self.max_frames, self.steps - 1) + 1)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def _frame_arrays(self, sample_index: int, frame: int):
        start, end = _frame_slice(self.offsets, sample_index, frame)
        return self.measurements[start:end], self.truth_ids[start:end]

    def build_example(self, sample_index: int, frame: int) -> AssociationExample:
        prev_start, prev_end = _frame_slice(self.offsets, sample_index, frame - 1)
        cur_start, cur_end = _frame_slice(self.offsets, sample_index, frame)
        prev_positions = self.measurements[prev_start:prev_end]
        prev_ids = self.truth_ids[prev_start:prev_end]
        cur_positions = self.measurements[cur_start:cur_end]
        cur_ids = self.truth_ids[cur_start:cur_end]

        # Rows are previous-frame true detections only (source id >= 0 selects a
        # row; clutter is not a candidate track).
        row_selector = prev_ids >= 0
        row_positions = prev_positions[row_selector]
        row_ids = prev_ids[row_selector]
        # Columns keep every measurement (true detections and clutter).
        col_positions = cur_positions
        col_ids = cur_ids

        rows = int(min(len(row_positions), self.pad_size))
        columns = int(min(len(col_positions), self.pad_size))

        row_mask = np.zeros(self.pad_size, dtype=bool)
        col_mask = np.zeros(self.pad_size, dtype=bool)
        row_mask[:rows] = True
        col_mask[:columns] = True
        target = np.full((self.pad_size, self.pad_size), PAD_TARGET, dtype=np.float32)
        if rows and columns:
            row_ids_used = row_ids[:rows]
            col_ids_used = col_ids[:columns]
            positive = (row_ids_used[:, None] == col_ids_used[None, :]) & (row_ids_used[:, None] >= 0)
            target[:rows, :columns] = positive.astype(np.float32)

        features = build_feature_tensor(
            row_positions[:rows],
            col_positions[:columns],
            self.pad_size,
            self.channels,
            self.x_bias,
            self.pad_value,
        )
        return AssociationExample(features, row_mask, col_mask, target, sample_index, frame)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample_index, frame = self.index[index]
        return self.build_example(sample_index, frame).to_tensors()


def split_sample_indices(sample_count: int, validation_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Deterministic sequence-level split used for checkpoint selection."""

    rng = np.random.default_rng(seed)
    order = rng.permutation(sample_count)
    validation_size = int(round(sample_count * validation_fraction))
    validation = sorted(int(value) for value in order[:validation_size])
    training = sorted(int(value) for value in order[validation_size:])
    return training, validation


def association_examples_stream(
    path: Path,
    pad_size: int = 40,
    channels: Tuple[str, ...] = ("x_rdf", "y_rdf", "range_rdf"),
    sample_indices: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
) -> Iterator[AssociationExample]:
    """Convenience streaming iterator over association examples (for tests/audits)."""

    dataset = PackedAssociationDataset(
        path, pad_size=pad_size, channels=channels, sample_indices=sample_indices, max_frames=max_frames
    )
    for sample_index, frame in dataset.index:
        yield dataset.build_example(sample_index, frame)
