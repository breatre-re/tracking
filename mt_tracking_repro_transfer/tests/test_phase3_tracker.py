"""Phase 3 filter, dynamic-score tracker, recovery and artifact tests."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.experiments.phase3_dynamic_score_evaluation import _dynamic_config, run_matrix
from src.evaluation.phase2_metrics import evaluate_scenario
from src.evaluation.phase3_metrics import recovery_metrics
from src.simulation.dataset import Detection, ScenarioConfig, SimulationData
from src.simulation.scenarios import resolve_scenarios, generate_scenario
from src.tracking.dynamic_score_tracker import CANDIDATE, CONFIRMED, DynamicScoreTracker, ScoreTrackerConfig
from src.tracking.filter import CartesianKalmanFilter, FilterConfig

TRACKER_CONFIG = Path("configs/phase3_dynamic_tracker.yaml")
PHASE2_CONFIG = Path("configs/phase2_scenarios.yaml")
DISTRIBUTIONS = ("baseline_distribution", "paper_assumption_v1")
SCENARIOS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")


def _straight_line_detections(steps=20, start=(10.0, 10.0), velocity=(1.0, 0.0), noise=0.0, seed=0, gap=None):
    rng = np.random.default_rng(seed)
    position = np.asarray(start, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    frames = []
    for time in range(steps):
        if gap is not None and gap[0] <= time < gap[1]:
            frames.append([])
        else:
            measurement = position + rng.normal(0.0, noise, 2) if noise > 0 else position.copy()
            frames.append([Detection(measurement, 0)])
        position = position + velocity
    return frames


def _simulation_from_positions(positions, config=None):
    positions = np.asarray(positions, dtype=float)
    steps = len(positions)
    truth = np.full((steps, 1, 4), np.nan)
    truth[:, 0, :2] = positions
    frames = [[Detection(positions[time], 0)] for time in range(steps)]
    config = config or ScenarioConfig(steps=steps, num_targets=1, seed=0)
    return SimulationData(truth, frames, np.array([0]), np.array([steps]), config)


class FilterTests(unittest.TestCase):
    def test_predict_advances_state_and_keeps_covariance_symmetric(self):
        kalman = CartesianKalmanFilter(np.array([0.0, 0.0]), FilterConfig())
        kalman.state = np.array([0.0, 0.0, 1.0, 2.0])
        kalman.predict()
        self.assertAlmostEqual(kalman.state[0], 1.0)
        self.assertAlmostEqual(kalman.state[1], 2.0)
        self.assertTrue(np.allclose(kalman.covariance, kalman.covariance.T))
        self.assertTrue(np.all(np.linalg.eigvalsh(kalman.covariance) > 0))

    def test_update_moves_toward_measurement_and_shrinks_uncertainty(self):
        kalman = CartesianKalmanFilter(np.array([0.0, 0.0]))
        before = kalman.covariance[0, 0]
        log_likelihood = kalman.update(np.array([1.0, -1.0]))
        self.assertTrue(np.isfinite(log_likelihood))
        self.assertLess(kalman.covariance[0, 0], before)
        self.assertGreater(kalman.state[0], 0.0)
        self.assertLess(kalman.state[1], 0.0)

    def test_repeated_updates_remain_positive_semi_definite(self):
        kalman = CartesianKalmanFilter(np.array([0.0, 0.0]))
        for step in range(40):
            kalman.predict()
            kalman.update(np.array([float(step), float(step % 3)]))
        self.assertTrue(np.allclose(kalman.covariance, kalman.covariance.T, atol=1e-9))
        self.assertTrue(np.all(np.linalg.eigvalsh(kalman.covariance) > -1e-9))

    def test_mahalanobis_matches_manual_quadratic_form(self):
        kalman = CartesianKalmanFilter(np.array([0.0, 0.0]))
        residual, covariance = kalman.innovation(np.array([3.0, 4.0]))
        expected = float(residual @ np.linalg.solve(covariance, residual))
        self.assertAlmostEqual(kalman.mahalanobis(np.array([3.0, 4.0])), expected)


class LifecycleTests(unittest.TestCase):
    def _tracker(self, **overrides):
        config = ScoreTrackerConfig(**overrides)
        return DynamicScoreTracker(config)

    def test_single_target_confirms_and_tracks(self):
        detections = _straight_line_detections(steps=20)
        tracker = self._tracker()
        result = tracker.run(detections)
        statuses = [len(frame) for frame in result.estimates_by_frame]
        self.assertEqual(statuses[-1], 1)
        self.assertGreater(tracker.diagnostics["confirmations"], 0)
        # Confirmed track should appear within a few frames of the first detection.
        self.assertLess(statuses.index(1), 6)
        self.assertEqual(tracker.diagnostics["final_confirmed"], 1.0)

    def test_lost_track_recovers_with_same_id(self):
        detections = _straight_line_detections(steps=30, gap=(10, 18))
        tracker = self._tracker()
        result = tracker.run(detections)
        before = result.estimates_by_frame[9]
        after = result.estimates_by_frame[18]
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 1)
        self.assertEqual(before[0].track_id, after[0].track_id)
        self.assertGreaterEqual(tracker.diagnostics["recoveries"], 1)

    def test_lost_track_deleted_after_expiry(self):
        detections = _straight_line_detections(steps=40, gap=(10, 40))
        tracker = self._tracker(lost_expiry=3)
        tracker.run(detections)
        self.assertGreaterEqual(tracker.diagnostics["deletions"], 1)
        self.assertEqual(len(tracker._tracks), 0)

    def test_missed_increment_modes_have_opposite_signs(self):
        documented = ScoreTrackerConfig(missed_increment_mode="documented")
        corrected = ScoreTrackerConfig(missed_increment_mode="likelihood_penalty")
        self.assertGreater(documented.missed_increment, 0.0)
        self.assertLess(corrected.missed_increment, 0.0)

    def test_corrected_mode_suppresses_clutter_candidates(self):
        rng = np.random.default_rng(7)
        frames = [[Detection(rng.uniform(0, 100, 2)) for _ in range(3)] for _ in range(30)]
        tracker = self._tracker(missed_increment_mode="likelihood_penalty")
        tracker.run(frames)
        self.assertEqual(tracker.diagnostics["max_concurrent_confirmed"], 0.0)

    def test_statuses_are_within_documented_set(self):
        detections = _straight_line_detections(steps=15)
        tracker = self._tracker()
        tracker.run(detections)
        for track in tracker._tracks:
            self.assertIn(track.status, (CANDIDATE, CONFIRMED))


class RepeatabilityAndIsolationTests(unittest.TestCase):
    def _spec(self, scenario="S1", distribution="baseline_distribution"):
        _, specs = resolve_scenarios(PHASE2_CONFIG, scenarios=[scenario], distributions=[distribution])
        return specs[0]

    def test_metrics_repeatable_runtime_finite(self):
        node = yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8"))["trackers"]["dynamic_score"]
        spec = self._spec("S1")
        simulation = generate_scenario(spec)
        detections = [[Detection(d.position.copy()) for d in frame] for frame in simulation.detections]

        def run_once():
            tracker = DynamicScoreTracker(_dynamic_config(node))
            import time

            start = time.perf_counter()
            result = tracker.run([[Detection(d.position.copy()) for d in frame] for frame in detections])
            runtime = time.perf_counter() - start
            return evaluate_scenario(simulation, result, 12.0), runtime

        first_metrics, first_runtime = run_once()
        second_metrics, second_runtime = run_once()
        self.assertEqual(first_metrics, second_metrics)
        for runtime in (first_runtime, second_runtime):
            self.assertGreaterEqual(runtime, 0.0)
            self.assertTrue(math.isfinite(runtime))

    def test_truth_labels_do_not_reach_tracker(self):
        node = yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8"))["trackers"]["dynamic_score"]
        spec = self._spec("S1")
        simulation = generate_scenario(spec)
        labelled = [[Detection(d.position.copy(), d.source_target_id) for d in frame] for frame in simulation.detections]
        unlabelled = [[Detection(d.position.copy()) for d in frame] for frame in simulation.detections]

        def signature(frames):
            tracker = DynamicScoreTracker(_dynamic_config(node))
            result = tracker.run(frames)
            return [
                [(e.track_id, tuple(np.round(e.state, 6))) for e in frame]
                for frame in result.estimates_by_frame
            ]

        self.assertEqual(signature(labelled), signature(unlabelled))

    def test_recovery_metrics_on_disappearance_window(self):
        _, specs = resolve_scenarios(PHASE2_CONFIG, scenarios=["S6"], distributions=["baseline_distribution"])
        spec = specs[0]
        node = yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8"))["trackers"]["dynamic_score_corrected"]
        simulation = generate_scenario(spec)
        tracker = DynamicScoreTracker(_dynamic_config(node))
        result = tracker.run([[Detection(d.position.copy()) for d in frame] for frame in simulation.detections])
        recovery = recovery_metrics(simulation, result, spec.disappearance, 12.0)
        self.assertEqual(recovery["opportunities"], 1.0)
        self.assertEqual(recovery["recovery_rate"], 1.0)


class Phase3MatrixTests(unittest.TestCase):
    def test_full_matrix_resolves_for_all_trackers(self):
        configuration = yaml.safe_load(TRACKER_CONFIG.read_text(encoding="utf-8"))
        self.assertIn("baseline", configuration["trackers"])
        self.assertIn("dynamic_score", configuration["trackers"])
        self.assertIn("dynamic_score_corrected", configuration["trackers"])

    def test_runner_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = run_matrix(
                TRACKER_CONFIG,
                root / "generated",
                root / "plots",
                root / "phase3_manifest.json",
                scenarios=["S0", "S6"],
                distributions=["baseline_distribution"],
                trackers=["baseline", "dynamic_score"],
            )
            self.assertEqual(manifest["case_count"], 4)
            for case in manifest["cases"]:
                self.assertTrue(Path(case["metrics_file"]).is_file())
                self.assertTrue(Path(case["plot_file"]).is_file())
                for key in ("matched_detections", "false_positives", "false_negatives", "id_switches",
                            "fragmentation", "position_rmse", "mota_like"):
                    self.assertIn(key, case["metrics"])
                self.assertIn("recovery_rate", case["recovery"])
                self.assertGreaterEqual(case["runtime_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
