"""Phase 3 additive dynamic-score tracker.

This tracker is added *alongside* the Phase 1/2 gated-greedy baseline in
``src/tracking/baseline_tracker.py``; the baseline is left untouched. It consumes the same
position-only ``Detection`` objects and emits the same ``TrackingResult`` shape,
so the existing Phase 2 evaluator applies unchanged.

Design (all numeric values are reproduction assumptions, exposed via
:class:`ScoreTrackerConfig`):

* 4D Cartesian recursive BLUE/Kalman filter per track (``src/tracking/filter.py``).
* Adaptive Mahalanobis association with a configurable normalised-distance gate
  and a configurable physical-distance cap, resolved as a one-to-one assignment.
* Additive score recursion from ``docs/plan/IMPLEMENTATION_SPEC.md`` Section 6::

      S_l(k) = lambda_s * S_l(k-1) + dS_l(k) - lambda_s**lambda_t * dS_l(k - lambda_t)

  with event increments
      * unmatched measurement -> new candidate:  dS = log(P_B / P_FA)
      * unmatched track (missed):                dS = log((1 - P_FA) / (1 - P_D))
      * matched track:                           dS = log p(z | prediction) - log(P_FA)
* Lifecycle states ``candidate`` / ``confirmed`` / ``lost`` / ``deleted``.
  Lost tracks stay association-eligible until ``lost_expiry`` so a target that
  reappears can keep the same ID (same-ID recovery).

Truth/source labels are never available here; an unmatched measurement cannot be
known to be clutter, so it is handled with the configured birth-vs-false-alarm
prior. Any counted recovery uses only the documented association increment plus
an optional ``reappearance_reward`` (default 0.0, i.e. no extra term).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.tracking.filter import CartesianKalmanFilter, FilterConfig
from src.tracking.baseline_tracker import Detection, TrackEstimate, TrackingResult


Array = np.ndarray

CANDIDATE = "candidate"
CONFIRMED = "confirmed"
LOST = "lost"


@dataclass
class ScoreTrackerConfig:
    """Filter, gate, score and lifecycle parameters (reproduction assumptions)."""

    # Filter noise / initialization.
    process_accel_std: float = 0.10
    measurement_noise_std: float = 1.5
    initial_position_var: float = 0.25
    initial_velocity_var: float = 1.0
    # Association gating.
    mahalanobis_gate: float = 9.21
    physical_gate: float = 10.0
    # Lost tracks coast for several frames, so their prediction drifts further;
    # a wider re-association gate supports same-ID recovery (engineering addition).
    lost_gate_scale: float = 2.5
    # Score recursion.
    score_decay: float = 0.95
    score_history_lag: int = 4
    score_confirm: float = 1.0
    p_birth: float = 0.05
    p_false_alarm: float = 0.02
    p_detect: float = 0.90
    # Lifecycle.
    max_missed: int = 5
    candidate_expiry: int = 6
    lost_expiry: int = 15
    reappearance_reward: float = 0.0
    report_lost: bool = False
    # Missed-detection increment sign.
    #   "documented"         -> log((1 - P_FA) / (1 - P_D))   (docs/plan/IMPLEMENTATION_SPEC.md)
    #   "likelihood_penalty" -> log((1 - P_D) / (1 - P_FA))   (sign-corrected penalty)
    # docs/plan/IMPLEMENTATION_SPEC.md documents the first form, but with P_D > P_FA it is
    # positive, which rewards unassociated tracks/candidates. The second form is
    # provided so the correction can be measured explicitly rather than silently
    # applied; the default stays faithful to the documented formula.
    missed_increment_mode: str = "documented"

    def filter_config(self) -> FilterConfig:
        return FilterConfig(
            process_accel_std=self.process_accel_std,
            measurement_noise_std=self.measurement_noise_std,
            initial_position_var=self.initial_position_var,
            initial_velocity_var=self.initial_velocity_var,
        )

    @property
    def birth_increment(self) -> float:
        return float(np.log(self.p_birth / self.p_false_alarm))

    @property
    def missed_increment(self) -> float:
        if self.missed_increment_mode == "likelihood_penalty":
            return float(np.log((1.0 - self.p_detect) / (1.0 - self.p_false_alarm)))
        # Documented increment: log((1 - P_FA) / (1 - P_D)).
        return float(np.log((1.0 - self.p_false_alarm) / (1.0 - self.p_detect)))


@dataclass
class _TrackState:
    track_id: int
    kalman: CartesianKalmanFilter
    status: str
    score: float
    age: int = 1
    hits: int = 1
    misses: int = 0
    lost_misses: int = 0
    # Score increments indexed by absolute frame, used for the lag term.
    increments: Dict[int, float] = field(default_factory=dict)


def _one_to_one_assignment(cost: Array, valid: Array) -> List[Tuple[int, int]]:
    """Resolve a gated one-to-one assignment for the given cost matrix."""

    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        large = float(cost.max()) + 1e6
        working = np.where(valid, cost, large)
        rows, columns = linear_sum_assignment(working)
        return [
            (int(row), int(column))
            for row, column in zip(rows, columns)
            if valid[row, column]
        ]
    except Exception:  # pragma: no cover - scipy fallback
        candidates = sorted(
            (float(cost[row, column]), row, column)
            for row in range(cost.shape[0])
            for column in range(cost.shape[1])
            if valid[row, column]
        )
        used_rows, used_columns = set(), set()
        matches: List[Tuple[int, int]] = []
        for _, row, column in candidates:
            if row in used_rows or column in used_columns:
                continue
            matches.append((row, column))
            used_rows.add(row)
            used_columns.add(column)
        return matches


class DynamicScoreTracker:
    """Additive dynamic-score multi-target tracker with recovery support."""

    def __init__(self, config: Optional[ScoreTrackerConfig] = None) -> None:
        self.config = config or ScoreTrackerConfig()
        self._next_id = 1
        self._time = -1
        self._tracks: List[_TrackState] = []
        self.diagnostics: Dict[str, object] = {}
        self._reset_counters()

    # -- scoring -----------------------------------------------------------
    def _reset_counters(self) -> None:
        self._confirmations = 0
        self._recoveries = 0
        self._lost_events = 0
        self._deletions = 0
        self._max_confirmed = 0
        self._confirmed_history: List[int] = []

    def _apply_score(self, track: _TrackState, frame: int, increment: float) -> None:
        lag = int(self.config.score_history_lag)
        lagged = track.increments.get(frame - lag, 0.0) if lag > 0 else 0.0
        decay = float(self.config.score_decay)
        track.score = decay * track.score + increment - (decay ** lag) * lagged
        track.increments[frame] = increment

    # -- association -------------------------------------------------------
    def _associate(
        self, measurements: Array
    ) -> Tuple[List[Tuple[int, int]], set, set]:
        track_count = len(self._tracks)
        measurement_count = len(measurements)
        if track_count == 0 or measurement_count == 0:
            return [], set(), set()

        cost = np.full((track_count, measurement_count), np.inf, dtype=float)
        for track_index, track in enumerate(self._tracks):
            scale = self.config.lost_gate_scale if track.status == LOST else 1.0
            mahalanobis_limit = self.config.mahalanobis_gate * scale
            physical_limit = self.config.physical_gate * scale
            for measurement_index in range(measurement_count):
                measurement = measurements[measurement_index]
                mahalanobis = track.kalman.mahalanobis(measurement)
                physical = float(np.linalg.norm(measurement - track.kalman.position))
                if mahalanobis <= mahalanobis_limit and physical <= physical_limit:
                    cost[track_index, measurement_index] = mahalanobis

        valid = np.isfinite(cost)
        pairs = _one_to_one_assignment(cost, valid)
        matched_tracks = {track_index for track_index, _ in pairs}
        matched_measurements = {measurement_index for _, measurement_index in pairs}
        return pairs, matched_tracks, matched_measurements

    # -- lifecycle helpers -------------------------------------------------
    def _new_candidate(self, measurement: Array, frame: int) -> None:
        track = _TrackState(
            track_id=self._next_id,
            kalman=CartesianKalmanFilter(measurement, self.config.filter_config()),
            status=CANDIDATE,
            score=0.0,
        )
        self._next_id += 1
        self._apply_score(track, frame, self.config.birth_increment)
        self._tracks.append(track)

    def _reportable(self, track: _TrackState) -> bool:
        if track.status == CONFIRMED:
            return True
        if track.status == LOST:
            return bool(self.config.report_lost)
        return False

    # -- main step ---------------------------------------------------------
    def step(self, detections: Sequence[Detection]) -> Tuple[List[TrackEstimate], List[Tuple[int, int]]]:
        self._time += 1
        frame = self._time
        measurements = np.asarray([detection.position for detection in detections], dtype=float)
        if measurements.ndim != 2:
            measurements = measurements.reshape((-1, 2))

        for track in self._tracks:
            track.kalman.predict()
            track.age += 1

        pairs, matched_tracks, matched_measurements = self._associate(measurements)
        # Stable IDs must be captured before tracks are created/removed below.
        id_pairs = [
            (self._tracks[track_index].track_id, measurement_index)
            for track_index, measurement_index in pairs
        ]

        # Matched tracks: BLUE update + association increment.
        for track_index, measurement_index in pairs:
            track = self._tracks[track_index]
            log_likelihood = track.kalman.update(measurements[measurement_index])
            increment = log_likelihood - float(np.log(self.config.p_false_alarm))
            recovered = track.status == LOST
            self._apply_score(track, frame, increment)
            track.hits += 1
            track.misses = 0
            track.lost_misses = 0
            if recovered:
                track.status = CONFIRMED
                track.score = max(track.score, self.config.score_confirm)
                if self.config.reappearance_reward:
                    self._apply_score(track, frame, float(self.config.reappearance_reward))
                self._recoveries += 1
            elif track.status == CANDIDATE and track.score >= self.config.score_confirm:
                track.status = CONFIRMED
                self._confirmations += 1

        # Unmatched tracks: missed-detection increment + lifecycle transitions.
        to_delete: List[_TrackState] = []
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks:
                continue
            track.misses += 1
            self._apply_score(track, frame, self.config.missed_increment)
            if track.status == CANDIDATE:
                if track.score < 0 or track.age > self.config.candidate_expiry:
                    to_delete.append(track)
            elif track.status == CONFIRMED:
                if track.score < 0 or track.misses > self.config.max_missed:
                    track.status = LOST
                    track.lost_misses = 0
                    self._lost_events += 1
            else:  # LOST
                track.lost_misses += 1
                if track.lost_misses > self.config.lost_expiry:
                    to_delete.append(track)

        for track in to_delete:
            self._deletions += 1
        if to_delete:
            deleted_ids = {id(track) for track in to_delete}
            self._tracks = [track for track in self._tracks if id(track) not in deleted_ids]

        # Unmatched measurements become new candidates.
        for measurement_index in range(len(measurements)):
            if measurement_index not in matched_measurements:
                self._new_candidate(measurements[measurement_index], frame)

        estimates = [
            TrackEstimate(
                track.track_id,
                track.kalman.state.copy(),
                track.status == CONFIRMED,
                track.hits,
                track.misses,
            )
            for track in self._tracks
            if self._reportable(track)
        ]
        confirmed_now = sum(1 for track in self._tracks if track.status == CONFIRMED)
        self._max_confirmed = max(self._max_confirmed, confirmed_now)
        self._confirmed_history.append(confirmed_now)
        return estimates, id_pairs

    def run(self, detections_by_frame: Sequence[Sequence[Detection]]) -> TrackingResult:
        self._reset_counters()
        estimates_by_frame: List[List[TrackEstimate]] = []
        associations_by_frame: List[List[Tuple[int, int]]] = []
        for detections in detections_by_frame:
            estimates, associations = self.step(detections)
            estimates_by_frame.append(estimates)
            associations_by_frame.append(associations)

        self.diagnostics = {
            "config": asdict(self.config),
            "confirmations": float(self._confirmations),
            "recoveries": float(self._recoveries),
            "lost_events": float(self._lost_events),
            "deletions": float(self._deletions),
            "max_concurrent_confirmed": float(self._max_confirmed),
            "final_confirmed": float(self._confirmed_history[-1]) if self._confirmed_history else 0.0,
            "final_active_tracks": float(len(self._tracks)),
            "births": float(self._next_id - 1),
        }
        return TrackingResult(estimates_by_frame, associations_by_frame)
