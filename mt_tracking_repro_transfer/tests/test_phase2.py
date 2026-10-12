"""Phase 2 deterministic regression, repeatability and label-isolation tests."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.experiments.phase2_baseline_evaluation import _tracker_detections, run_matrix, run_tracker
from src.evaluation.phase2_metrics import evaluate_scenario
from src.simulation.dataset import load_dataset_sample
from src.simulation.scenarios import generate_scenario, resolve_scenarios

CONFIG_PATH = Path("configs/phase2_scenarios.yaml")
DISTRIBUTIONS = ("baseline_distribution", "paper_assumption_v1")
SCENARIOS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")

METRIC_KEYS = (
    "frames",
    "truth_object_frames",
    "truth_objects_per_frame",
    "emitted_estimates",
    "matched_detections",
    "false_negatives",
    "false_positives",
    "id_switches",
    "fragmentation",
    "position_rmse",
    "mean_position_error",
    "mota_like",
)


def _spec(name: str, distribution: str):
    _, specs = resolve_scenarios(CONFIG_PATH, scenarios=[name], distributions=[distribution])
    return specs[0]


def _result_signature(result) -> list:
    signature = []
    for estimates in result.estimates_by_frame:
        signature.append(
            [(estimate.track_id, tuple(np.asarray(estimate.state, dtype=float))) for estimate in estimates]
        )
    return signature


class ScenarioMatrixTests(unittest.TestCase):
    def test_full_matrix_resolves(self):
        _, specs = resolve_scenarios(CONFIG_PATH)
        pairs = [(spec.distribution, spec.name) for spec in specs]
        self.assertEqual(len(pairs), len(DISTRIBUTIONS) * len(SCENARIOS))
        for distribution in DISTRIBUTIONS:
            for scenario in SCENARIOS:
                self.assertIn((distribution, scenario), pairs)

    def test_generation_is_deterministic(self):
        for name, distribution in (("S0", "baseline_distribution"), ("S7", "paper_assumption_v1")):
            spec = _spec(name, distribution)
            first = generate_scenario(spec)
            second = generate_scenario(spec)
            self.assertTrue(np.array_equal(first.true_states, second.true_states, equal_nan=True))
            self.assertEqual(len(first.detections), len(second.detections))
            for frame_a, frame_b in zip(first.detections, second.detections):
                self.assertEqual(len(frame_a), len(frame_b))
                for detection_a, detection_b in zip(frame_a, frame_b):
                    self.assertTrue(np.array_equal(detection_a.position, detection_b.position))
                    self.assertEqual(detection_a.source_target_id, detection_b.source_target_id)

    def test_s0_is_noise_free_single_target(self):
        spec = _spec("S0", "baseline_distribution")
        self.assertEqual(spec.num_targets, 1)
        self.assertEqual(spec.clutter_rate, 0.0)
        self.assertEqual(spec.measurement_std, 0.0)
        self.assertEqual(spec.process_accel_std, 0.0)
        self.assertEqual(spec.detection_probability, 1.0)
        simulation = generate_scenario(spec)
        for frame in simulation.detections:
            self.assertEqual(len(frame), 1)

    def test_s1_uses_non_crossing_parallel_motion(self):
        spec = _spec("S1", "baseline_distribution")
        self.assertEqual(spec.motion, "parallel")
        simulation = generate_scenario(spec)
        lane_heights = simulation.true_states[0, :, 1]
        self.assertEqual(len(np.unique(lane_heights)), spec.num_targets)

    def test_s5_lifetimes_match_requested_windows(self):
        spec = _spec("S5", "baseline_distribution")
        simulation = generate_scenario(spec)
        for (birth, death), actual_birth, actual_death in zip(
            spec.lifetimes, simulation.birth_times, simulation.death_times
        ):
            self.assertEqual(int(actual_birth), birth)
            self.assertEqual(int(actual_death), death)
        self.assertGreater(len({birth for birth, _ in spec.lifetimes}), 1)

    def test_s6_has_a_detection_gap_without_losing_truth(self):
        spec = _spec("S6", "baseline_distribution")
        target, start, end = spec.disappearance[0]
        simulation = generate_scenario(spec)
        for time in range(start, end):
            sources = {
                detection.source_target_id
                for detection in simulation.detections[time]
                if detection.source_target_id is not None
            }
            self.assertNotIn(target, sources)
            self.assertTrue(np.isfinite(simulation.true_states[time, target, 0]))
        gap_free_frame = end
        sources = {
            detection.source_target_id
            for detection in simulation.detections[gap_free_frame]
            if detection.source_target_id is not None
        }
        self.assertIn(target, sources)

    def test_s7_is_high_target_high_clutter(self):
        for distribution in DISTRIBUTIONS:
            spec = _spec("S7", distribution)
            self.assertEqual(spec.num_targets, 25)
            self.assertEqual(spec.clutter_rate, 20.0)

    def test_paper_distribution_keeps_high_clutter_default(self):
        for name in ("S1", "S2", "S3", "S5", "S6"):
            spec = _spec(name, "paper_assumption_v1")
            self.assertEqual(spec.clutter_rate, 20.0)


class MetricCompletenessTests(unittest.TestCase):
    def test_metrics_are_complete_and_finite(self):
        spec = _spec("S1", "baseline_distribution")
        simulation = generate_scenario(spec)
        result, runtime = run_tracker(simulation)
        metrics = evaluate_scenario(simulation, result, match_gate=12.0)
        for key in METRIC_KEYS:
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]), key)
        self.assertGreaterEqual(runtime, 0.0)
        self.assertTrue(math.isfinite(runtime))

    def test_metrics_repeatable_but_runtime_only_finite(self):
        spec = _spec("S2", "paper_assumption_v1")
        first_sim = generate_scenario(spec)
        second_sim = generate_scenario(spec)
        first_result, first_runtime = run_tracker(first_sim)
        second_result, second_runtime = run_tracker(second_sim)
        first_metrics = evaluate_scenario(first_sim, first_result, match_gate=12.0)
        second_metrics = evaluate_scenario(second_sim, second_result, match_gate=12.0)
        self.assertEqual(first_metrics, second_metrics)
        for runtime in (first_runtime, second_runtime):
            self.assertGreaterEqual(runtime, 0.0)
            self.assertTrue(math.isfinite(runtime))

    def test_s0_matches_every_frame(self):
        spec = _spec("S0", "baseline_distribution")
        simulation = generate_scenario(spec)
        result, _ = run_tracker(simulation)
        metrics = evaluate_scenario(simulation, result, match_gate=12.0)
        self.assertEqual(metrics["false_negatives"], 0.0)
        self.assertEqual(metrics["false_positives"], 0.0)
        self.assertEqual(metrics["id_switches"], 0.0)
        self.assertEqual(metrics["mota_like"], 1.0)


class LabelIsolationTests(unittest.TestCase):
    def test_tracker_input_carries_no_source_labels(self):
        simulation = generate_scenario(_spec("S4", "baseline_distribution"))
        detections = _tracker_detections(simulation)
        for frame in detections:
            for detection in frame:
                self.assertIsNone(detection.source_target_id)

    def test_tracker_output_independent_of_source_labels(self):
        spec = _spec("S1", "baseline_distribution")
        simulation = generate_scenario(spec)
        baseline_result, _ = run_tracker(simulation)

        rng = np.random.default_rng(1234)
        for frame in simulation.detections:
            for detection in frame:
                detection.source_target_id = int(rng.integers(0, 100))
        relabelled_result, _ = run_tracker(simulation)

        self.assertEqual(_result_signature(baseline_result), _result_signature(relabelled_result))


class Phase2ArtifactTests(unittest.TestCase):
    def test_manifest_and_artifacts_are_written(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = run_matrix(
                CONFIG_PATH,
                root / "generated",
                root / "plots",
                root / "phase2_manifest.json",
                scenarios=["S0", "S1"],
                distributions=["baseline_distribution"],
            )
            self.assertEqual(manifest["case_count"], 2)
            for case in manifest["cases"]:
                self.assertTrue(Path(case["metrics_file"]).is_file())
                self.assertTrue(Path(case["plot_file"]).is_file())
                payload = json.loads(Path(case["metrics_file"]).read_text(encoding="utf-8"))
                for key in METRIC_KEYS:
                    self.assertIn(key, payload["metrics"])
                self.assertEqual(payload["resolved_config"]["seed"], case["resolved_config"]["seed"])


class Phase1CompatibilityTests(unittest.TestCase):
    def test_phase1_schema1_dataset_loads_and_tracks(self):
        dataset_path = Path("data/datasets/phase1/smoke/smoke_dataset.npz")
        if not dataset_path.is_file():
            self.skipTest("Phase 1 smoke dataset not present")
        sample = load_dataset_sample(dataset_path, sample_index=0)
        result, runtime = run_tracker(sample.simulation)
        metrics = evaluate_scenario(sample.simulation, result, match_gate=12.0)
        self.assertGreater(metrics["frames"], 0)
        self.assertTrue(math.isfinite(metrics["mota_like"]))
        self.assertGreaterEqual(runtime, 0.0)

    def test_phase1_5_schema2_dataset_loads_and_tracks(self):
        dataset_path = Path("data/datasets/phase1_5/baseline_distribution/test/test_dataset.npz")
        if not dataset_path.is_file():
            self.skipTest("Phase 1.5 baseline test dataset not present")
        for sample_index in (0, 1):
            sample = load_dataset_sample(dataset_path, sample_index=sample_index)
            result, runtime = run_tracker(sample.simulation)
            metrics = evaluate_scenario(sample.simulation, result, match_gate=12.0)
            self.assertGreater(metrics["frames"], 0)
            self.assertTrue(math.isfinite(metrics["mota_like"]))
            self.assertGreaterEqual(runtime, 0.0)


if __name__ == "__main__":
    unittest.main()
