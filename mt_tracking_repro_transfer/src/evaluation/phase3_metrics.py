"""Phase 3 recovery and lifecycle evaluation helpers.

Position/identity quality metrics (FP, FN, ID switches, fragmentation, RMSE,
MOTA-like) come from the unchanged Phase 2 evaluator. This module adds the
Phase 3-specific offline recovery measurement for configured disappearance
windows.

Recovery definition: for a configured disappearance window ``(target, start,
end)`` the truth target is active throughout, but the generator forces no
detections in ``[start, end)``. A recovery succeeds when the same estimated track
ID matches that same ground-truth target in the last matched frame before the gap
and in the first matched frame at/after the gap. A window is an *opportunity*
only if the target was matched to a track before the gap.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.simulation.dataset import SimulationData
from src.tracking.baseline_tracker import TrackingResult


Array = np.ndarray


def _greedy_match(truth_positions: Array, estimate_positions: Array, gate: float) -> List[Tuple[int, int]]:
    """Framewise greedy position match; mirrors the Phase 2 matcher semantics."""

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
    matches: List[Tuple[int, int]] = []
    for _, target_index, estimate_index in candidates:
        if target_index in used_truth or estimate_index in used_estimates:
            continue
        matches.append((target_index, estimate_index))
        used_truth.add(target_index)
        used_estimates.add(estimate_index)
    return matches


def matched_track_series(
    simulation: SimulationData,
    result: TrackingResult,
    match_gate: float,
) -> Dict[int, Dict[int, int]]:
    """Return ``{truth_target: {frame: track_id}}`` for all matched frames."""

    series: Dict[int, Dict[int, int]] = {}
    for frame, estimates in enumerate(result.estimates_by_frame):
        truth_frame = simulation.true_states[frame]
        truth_ids = np.flatnonzero(np.isfinite(truth_frame[:, 0]))
        truth_positions = np.asarray(truth_frame[truth_ids, :2], dtype=float)
        estimate_positions = np.asarray([estimate.state[:2] for estimate in estimates], dtype=float)
        for target_index, estimate_index in _greedy_match(truth_positions, estimate_positions, match_gate):
            truth_id = int(truth_ids[target_index])
            series.setdefault(truth_id, {})[frame] = estimates[estimate_index].track_id
    return series


def recovery_metrics(
    simulation: SimulationData,
    result: TrackingResult,
    windows: Sequence[Tuple[int, int, int]],
    match_gate: float = 12.0,
) -> Dict[str, object]:
    """Compute same-ID recovery statistics for the configured windows."""

    series = matched_track_series(simulation, result, match_gate)
    details: List[Dict[str, object]] = []
    opportunities = 0
    successes = 0

    for target, start, end in windows:
        target_series = series.get(int(target), {})
        pre_frames = [frame for frame in target_series if frame < start]
        post_frames = [frame for frame in target_series if frame >= end]
        pre_id = target_series[max(pre_frames)] if pre_frames else None
        post_id = target_series[min(post_frames)] if post_frames else None
        recovered = pre_id is not None and post_id is not None and pre_id == post_id
        if pre_id is not None:
            opportunities += 1
            if recovered:
                successes += 1
        details.append(
            {
                "target": int(target),
                "start": int(start),
                "end": int(end),
                "pre_track_id": None if pre_id is None else int(pre_id),
                "post_track_id": None if post_id is None else int(post_id),
                "recovered": bool(recovered),
                "eligible": pre_id is not None,
            }
        )

    return {
        "opportunities": float(opportunities),
        "successes": float(successes),
        # None (JSON null) when there is no eligible gap; avoids non-standard NaN.
        "recovery_rate": (float(successes) / opportunities) if opportunities else None,
        "details": details,
    }
