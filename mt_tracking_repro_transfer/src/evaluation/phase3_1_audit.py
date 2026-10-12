"""Phase 3.1 offline audit diagnostics.

Ground-truth source labels are never available to online tracking; they are used
only here, offline, to classify what the tracker did. This module adds no
tracker behaviour and does not modify any existing evaluator.

It reports, per run:

* total associations and how many of them landed on a true detection vs a
  clutter measurement;
* unmatched measurements that were true detections vs clutter (each unmatched
  measurement spawns one new candidate in the dynamic-score tracker);
* therefore an estimate of how many candidates were born from clutter, which is
  the mechanism by which a positive missed-detection increment promotes clutter
  tracks.

Classification uses ``Detection.source_target_id`` from the generator, which is
``None`` for clutter and an integer target index for a true detection.
"""

from __future__ import annotations

from typing import Dict

from src.simulation.dataset import SimulationData
from src.tracking.baseline_tracker import TrackingResult


def association_outcome_diagnostics(
    simulation: SimulationData, result: TrackingResult
) -> Dict[str, float]:
    """Classify associations and unmatched measurements against truth labels."""

    associations_to_true = 0
    associations_to_clutter = 0
    unmatched_true = 0
    unmatched_clutter = 0
    total_measurements = 0

    for frame, detections in enumerate(simulation.detections):
        labels = [detection.source_target_id for detection in detections]
        total_measurements += len(labels)
        matched_indices = set()
        for _, measurement_index in result.associations_by_frame[frame]:
            matched_indices.add(measurement_index)
            if labels[measurement_index] is None:
                associations_to_clutter += 1
            else:
                associations_to_true += 1
        for measurement_index, label in enumerate(labels):
            if measurement_index in matched_indices:
                continue
            if label is None:
                unmatched_clutter += 1
            else:
                unmatched_true += 1

    total_associations = associations_to_true + associations_to_clutter
    return {
        "total_measurements": float(total_measurements),
        "total_associations": float(total_associations),
        "associations_to_true": float(associations_to_true),
        "associations_to_clutter": float(associations_to_clutter),
        "unmatched_true_detections": float(unmatched_true),
        "unmatched_clutter": float(unmatched_clutter),
        # Each unmatched measurement becomes one candidate; these approximate
        # how many candidates were born from clutter vs from a missed detection.
        "candidates_born_from_true": float(unmatched_true),
        "candidates_born_from_clutter": float(unmatched_clutter),
    }
