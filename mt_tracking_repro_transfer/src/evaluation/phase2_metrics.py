"""Phase 2 multi-target tracking evaluation metrics.

The evaluator is deliberately independent from the tracker: it only consumes the
ground-truth ``SimulationData`` (labels live here, never in the tracker input)
and the tracker's ``TrackingResult``. It performs a framewise greedy
truth-to-estimate position match and derives the Phase 2 metric set.

Metric definitions (per scenario, summed/aggregated over all frames):

* truth object-frames: one count per (frame, target) pair where the target is
  active (finite truth position) in that frame.
* matched detections: matched object-frames, i.e. an active truth target matched
  to a tracker estimate within ``match_gate`` metres.
* false negatives = truth object-frames - matched detections.
* false positives = emitted tracker estimates - matched detections.
* ID switches: number of times a matched truth target changes its assigned track
  ID between two consecutive frames in which it is matched.
* fragmentation: for each truth target, take the sorted frames in which it is
  matched, group them into contiguous runs, and add ``runs - 1``. This counts how
  often an established target track is interrupted.
* position RMSE = sqrt(mean(d**2)) over matched pairs, where ``d`` is the matched
  truth-to-estimate Euclidean distance (metres).
* MOTA-like = 1 - (FN + FP + ID switches) / truth object-frames.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.simulation.dataset import SimulationData
from src.tracking.baseline_tracker import TrackingResult


Array = np.ndarray


def _greedy_match(truth_positions: Array, estimate_positions: Array, gate: float) -> List[Tuple[int, int, float]]:
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
    matches: List[Tuple[int, int, float]] = []
    for distance, target_index, estimate_index in candidates:
        if target_index in used_truth or estimate_index in used_estimates:
            continue
        matches.append((target_index, estimate_index, distance))
        used_truth.add(target_index)
        used_estimates.add(estimate_index)
    return matches


def _fragmentation(matched_frames_by_target: Dict[int, List[int]]) -> int:
    fragmentation = 0
    for frames in matched_frames_by_target.values():
        if not frames:
            continue
        runs = 1
        for previous, current in zip(frames, frames[1:]):
            if current != previous + 1:
                runs += 1
        fragmentation += runs - 1
    return fragmentation


def evaluate_scenario(
    simulation: SimulationData,
    result: TrackingResult,
    match_gate: float = 12.0,
) -> Dict[str, float]:
    """Compute the Phase 2 metric set for one scenario run."""

    total_truth = 0
    total_estimates = 0
    matched_detections = 0
    id_switches = 0
    position_errors: List[float] = []
    last_track_for_truth: Dict[int, int] = {}
    matched_frames_by_target: Dict[int, List[int]] = {}

    for time, estimates in enumerate(result.estimates_by_frame):
        truth_frame = simulation.true_states[time]
        truth_ids = np.flatnonzero(np.isfinite(truth_frame[:, 0]))
        truth_positions = np.asarray(truth_frame[truth_ids, :2], dtype=float)
        estimate_positions = np.asarray([estimate.state[:2] for estimate in estimates], dtype=float)
        total_truth += len(truth_ids)
        total_estimates += len(estimates)

        matches = _greedy_match(truth_positions, estimate_positions, match_gate)
        matched_detections += len(matches)
        for target_index, estimate_index, distance in matches:
            truth_id = int(truth_ids[target_index])
            track_id = estimates[estimate_index].track_id
            position_errors.append(distance)
            matched_frames_by_target.setdefault(truth_id, []).append(time)
            previous = last_track_for_truth.get(truth_id)
            if previous is not None and previous != track_id:
                id_switches += 1
            last_track_for_truth[truth_id] = track_id

    false_negatives = total_truth - matched_detections
    false_positives = total_estimates - matched_detections
    frames = max(len(result.estimates_by_frame), 1)
    rmse = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else float("nan")
    mean_error = float(np.mean(position_errors)) if position_errors else float("nan")
    mota_like = 1.0 - (false_negatives + false_positives + id_switches) / max(total_truth, 1)

    return {
        "frames": float(len(result.estimates_by_frame)),
        "truth_object_frames": float(total_truth),
        "truth_objects_per_frame": float(total_truth / frames),
        "emitted_estimates": float(total_estimates),
        "matched_detections": float(matched_detections),
        "false_negatives": float(false_negatives),
        "false_positives": float(false_positives),
        "id_switches": float(id_switches),
        "fragmentation": float(_fragmentation(matched_frames_by_target)),
        "position_rmse": rmse,
        "mean_position_error": mean_error,
        "mota_like": float(mota_like),
    }
