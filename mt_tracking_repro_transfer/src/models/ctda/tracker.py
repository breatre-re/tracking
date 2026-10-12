"""Phase 4 CTDA + dynamic-score tracker integration.

``CTDADynamicScoreTracker`` subclasses the Phase 3 ``DynamicScoreTracker`` and
overrides **only** the association step to use a CTDA network instead of the
Mahalanobis gate. Phase 3 code is untouched; the BLUE filter, score recursion
(variant B, documented) and lifecycle all come from the parent class.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from src.models.ctda.adapter import CTDAAssociation
from src.tracking.dynamic_score_tracker import DynamicScoreTracker, ScoreTrackerConfig
from src.tracking.baseline_tracker import Detection


Array = np.ndarray


class CTDADynamicScoreTracker(DynamicScoreTracker):
    """Dynamic-score tracker whose association is a CTDA network."""

    def __init__(self, config: Optional[ScoreTrackerConfig] = None, association: Optional[CTDAAssociation] = None) -> None:
        super().__init__(config)
        if association is None:
            raise ValueError("CTDADynamicScoreTracker requires a CTDAAssociation")
        self.association = association
        self.last_association_pairs: List[Tuple[int, int]] = []

    def _associate(
        self, measurements: Array
    ) -> Tuple[List[Tuple[int, int]], Set[int], Set[int]]:
        tracks = len(self._tracks)
        measurement_count = len(measurements)
        if tracks == 0 or measurement_count == 0:
            return [], set(), set()

        predicted_positions = np.asarray([track.kalman.position for track in self._tracks], dtype=float)
        # CTDA consumes positions only; the gate argument is passed for interface
        # compatibility (the network's own threshold governs acceptance).
        pairs = self.association.associate(predicted_positions, measurements, self.config.physical_gate)
        matched_tracks = {track_index for track_index, _ in pairs}
        matched_measurements = {measurement_index for _, measurement_index in pairs}
        self.last_association_pairs = list(pairs)
        return pairs, matched_tracks, matched_measurements
