"""A small, NumPy-only 2D multi-target tracking reproduction prototype.

The prototype mirrors the part of the paper that starts with noisy point
measurements (the detector is intentionally out of scope):

    simulated truth -> noisy detections -> prediction/association
    -> track birth/confirmation/maintenance/deletion -> metrics

The association layer is deliberately isolated behind ``AssociationModel``.
``CTDAAssociationInterface`` is a drop-in seam for a future Transformer model;
the default runnable implementation is the gated greedy nearest-neighbour
baseline.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from src.simulation.dataset import (
    Detection,
    ScenarioConfig,
    SimulationData,
    load_dataset_sample,
    simulate_scenario,
)


Array = np.ndarray


@dataclass
class TrackEstimate:
    track_id: int
    state: Array
    confirmed: bool
    hits: int
    misses: int


@dataclass
class TrackingResult:
    estimates_by_frame: List[List[TrackEstimate]]
    associations_by_frame: List[List[Tuple[int, int]]]


def _constant_velocity_matrix() -> Array:
    return np.array(
        [[1.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 1.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


class AssociationModel:
    """Interface consumed by ``MultiTargetTracker``.

    A future CTDA implementation can replace ``associate`` while leaving
    prediction, lifecycle management, IDs and evaluation unchanged.
    """

    def associate(self, predicted_positions: Array, measurements: Array, gate: float) -> List[Tuple[int, int]]:
        raise NotImplementedError


class GatedGreedyAssociation(AssociationModel):
    """Simple, dependency-free baseline association."""

    def associate(self, predicted_positions: Array, measurements: Array, gate: float) -> List[Tuple[int, int]]:
        if len(predicted_positions) == 0 or len(measurements) == 0:
            return []
        distances = np.linalg.norm(predicted_positions[:, None, :] - measurements[None, :, :], axis=2)
        candidates = [
            (float(distances[track_index, measurement_index]), track_index, measurement_index)
            for track_index in range(len(predicted_positions))
            for measurement_index in range(len(measurements))
            if distances[track_index, measurement_index] <= gate
        ]
        candidates.sort(key=lambda item: item[0])
        used_tracks, used_measurements = set(), set()
        result: List[Tuple[int, int]] = []
        for _, track_index, measurement_index in candidates:
            if track_index not in used_tracks and measurement_index not in used_measurements:
                result.append((track_index, measurement_index))
                used_tracks.add(track_index)
                used_measurements.add(measurement_index)
        return result


class CTDAAssociationInterface(GatedGreedyAssociation):
    """Placeholder seam for the paper's Transformer-based CTDA association.

    It intentionally behaves like the baseline so the full prototype remains
    runnable. Replace ``associate`` with CTDA score-matrix inference later.
    Expected future output: a one-to-one list of (track_index, measurement_index)
    pairs, optionally derived from a learned association probability matrix.
    """

    pass


@dataclass
class _InternalTrack:
    track_id: int
    state: Array
    covariance: Array
    age: int = 1
    hits: int = 1
    misses: int = 0
    confirmed: bool = False


class MultiTargetTracker:
    """Constant-velocity tracker with gated association and lifecycle logic."""

    def __init__(
        self,
        association_model: Optional[AssociationModel] = None,
        gate: float = 9.0,
        min_hits: int = 2,
        max_missed: int = 5,
    ) -> None:
        self.association_model = association_model or GatedGreedyAssociation()
        self.gate = gate
        self.min_hits = min_hits
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: List[_InternalTrack] = []
        self._f = _constant_velocity_matrix()
        self._q = np.diag([0.15, 0.15, 0.08, 0.08])

    def _predict(self) -> None:
        for track in self._tracks:
            track.state = self._f @ track.state
            track.covariance = self._f @ track.covariance @ self._f.T + self._q
            track.age += 1

    def _new_track(self, position: Array) -> _InternalTrack:
        state = np.array([position[0], position[1], 0.0, 0.0], dtype=float)
        covariance = np.diag([4.0, 4.0, 9.0, 9.0])
        track = _InternalTrack(self._next_id, state, covariance)
        self._next_id += 1
        self._tracks.append(track)
        return track

    def step(self, detections: Sequence[Detection]) -> Tuple[List[TrackEstimate], List[Tuple[int, int]]]:
        self._predict()
        measurements = np.array([d.position for d in detections], dtype=float)
        predicted_positions = np.array([track.state[:2] for track in self._tracks], dtype=float)
        pairs = self.association_model.associate(predicted_positions, measurements, self.gate)
        # Convert transient list indices to stable IDs before new tracks are
        # created or stale tracks are removed below.
        id_pairs = [(self._tracks[track_index].track_id, measurement_index) for track_index, measurement_index in pairs]
        matched_tracks = {track_index for track_index, _ in pairs}
        matched_measurements = {measurement_index for _, measurement_index in pairs}

        for track_index, measurement_index in pairs:
            track = self._tracks[track_index]
            measurement = measurements[measurement_index]
            innovation = measurement - track.state[:2]
            track.state[:2] += 0.65 * innovation
            track.state[2:] += 0.18 * innovation
            track.hits += 1
            track.misses = 0
            if track.hits >= self.min_hits:
                track.confirmed = True

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.misses += 1

        for measurement_index, measurement in enumerate(measurements):
            if measurement_index not in matched_measurements:
                self._new_track(measurement)

        self._tracks = [
            track for track in self._tracks
            if track.misses <= (self.max_missed if track.confirmed else 1)
        ]
        estimates = [
            TrackEstimate(track.track_id, track.state.copy(), track.confirmed, track.hits, track.misses)
            for track in self._tracks
            if track.confirmed or track.misses == 0
        ]
        return estimates, id_pairs

    def run(self, detections_by_frame: Sequence[Sequence[Detection]]) -> TrackingResult:
        estimates_by_frame: List[List[TrackEstimate]] = []
        associations_by_frame: List[List[Tuple[int, int]]] = []
        for detections in detections_by_frame:
            estimates, associations = self.step(detections)
            estimates_by_frame.append(estimates)
            associations_by_frame.append(associations)
        return TrackingResult(estimates_by_frame, associations_by_frame)


def _greedy_truth_match(truth_positions: Array, estimate_positions: Array, gate: float) -> List[Tuple[int, int, float]]:
    if len(truth_positions) == 0 or len(estimate_positions) == 0:
        return []
    distances = np.linalg.norm(truth_positions[:, None, :] - estimate_positions[None, :, :], axis=2)
    candidates = [
        (float(distances[target_index, estimate_index]), target_index, estimate_index)
        for target_index in range(len(truth_positions))
        for estimate_index in range(len(estimate_positions))
        if distances[target_index, estimate_index] <= gate
    ]
    candidates.sort()
    used_truth, used_estimates = set(), set()
    result = []
    for distance, target_index, estimate_index in candidates:
        if target_index not in used_truth and estimate_index not in used_estimates:
            result.append((target_index, estimate_index, distance))
            used_truth.add(target_index)
            used_estimates.add(estimate_index)
    return result


def evaluate_tracking(simulation: SimulationData, result: TrackingResult, match_gate: float = 12.0) -> Dict[str, float]:
    """Compute basic detection/track quality metrics and ID switches."""

    total_truth = false_positives = false_negatives = 0
    position_errors: List[float] = []
    id_switches = 0
    last_track_for_truth: Dict[int, int] = {}
    matched_frames = 0

    for time, estimates in enumerate(result.estimates_by_frame):
        truth_frame = simulation.true_states[time]
        truth_ids = np.flatnonzero(np.isfinite(truth_frame[:, 0]))
        truth_positions = truth_frame[truth_ids, :2]
        estimate_positions = np.array([estimate.state[:2] for estimate in estimates], dtype=float)
        matches = _greedy_truth_match(truth_positions, estimate_positions, match_gate)
        total_truth += len(truth_ids)
        false_negatives += len(truth_ids) - len(matches)
        false_positives += len(estimates) - len(matches)
        if matches:
            matched_frames += len(matches)
        for _, estimate_index, distance in matches:
            truth_id = int(truth_ids[_])
            track_id = estimates[estimate_index].track_id
            position_errors.append(distance)
            previous = last_track_for_truth.get(truth_id)
            if previous is not None and previous != track_id:
                id_switches += 1
            last_track_for_truth[truth_id] = track_id

    mota = 1.0 - (false_negatives + false_positives + id_switches) / max(total_truth, 1)
    return {
        "frames": float(len(result.estimates_by_frame)),
        "ground_truth_objects_per_frame": float(total_truth / max(len(result.estimates_by_frame), 1)),
        "matched_object_frames": float(matched_frames),
        "false_negatives": float(false_negatives),
        "false_positives": float(false_positives),
        "id_switches": float(id_switches),
        "mean_position_error": float(np.mean(position_errors)) if position_errors else float("nan"),
        "mota_like": float(mota),
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run_demo(args: argparse.Namespace) -> Dict[str, object]:
    if args.dataset:
        dataset_sample = load_dataset_sample(Path(args.dataset), args.sample_index)
        simulation = dataset_sample.simulation
        config = simulation.config
    else:
        config = ScenarioConfig(
            steps=args.steps,
            num_targets=args.targets,
            seed=args.seed,
            detection_probability=args.detection_probability,
            measurement_std=args.measurement_std,
            clutter_rate=args.clutter_rate,
        )
        simulation = simulate_scenario(config)
    model: AssociationModel = (
        CTDAAssociationInterface() if args.association == "ctda-interface" else GatedGreedyAssociation()
    )
    tracker = MultiTargetTracker(model, gate=args.gate, min_hits=args.min_hits, max_missed=args.max_missed)
    tracker_detections = [
        [Detection(detection.position.copy()) for detection in frame]
        for frame in simulation.detections
    ]
    result = tracker.run(tracker_detections)
    metrics = evaluate_tracking(simulation, result)
    summary: Dict[str, object] = {
        "config": asdict(config),
        "association": args.association,
        "data_source": args.dataset or "on-the-fly-simulation",
        "metrics": metrics,
        "notes": [
            "CTDA interface currently delegates to the gated greedy baseline.",
            "Detection source labels are used only by offline evaluation, never by the tracker.",
        ],
    }
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--targets", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--detection-probability", type=float, default=0.90)
    parser.add_argument("--measurement-std", type=float, default=1.5)
    parser.add_argument("--clutter-rate", type=float, default=1.5)
    parser.add_argument("--gate", type=float, default=9.0)
    parser.add_argument("--min-hits", type=int, default=2)
    parser.add_argument("--max-missed", type=int, default=5)
    parser.add_argument("--association", choices=["baseline", "ctda-interface"], default="baseline")
    parser.add_argument("--dataset", type=str, default="", help="Packed Phase 1 dataset NPZ")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index in --dataset")
    parser.add_argument("--output", type=str, default="", help="Optional JSON summary output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_demo(args)
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
