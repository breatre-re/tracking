"""Phase 3.1 score-direction audit tests.

Covers: missed-increment sign directions (A negative, B positive), the variant
parity contract (only the sign mode differs), input/config parity, offline
association-outcome diagnostics, label isolation, determinism and artifact
emission.
"""

import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from src.experiments.phase3_1_score_direction_audit import (
    _input_hashes,
    _run_case,
    _variant_node,
    build_parser,
    run_matrix,
)
from src.experiments.phase3_dynamic_score_evaluation import _dynamic_config
from src.evaluation.phase3_1_audit import association_outcome_diagnostics
from src.simulation.dataset import Detection, ScenarioConfig, SimulationData
from src.simulation.scenarios import generate_scenario, resolve_scenarios
from src.tracking.dynamic_score_tracker import DynamicScoreTracker, ScoreTrackerConfig
from src.tracking.baseline_tracker import TrackingResult, TrackEstimate

CONFIG_PATH = Path("configs/phase3_1_score_audit.yaml")
PHASE2_CONFIG = Path("configs/phase2_scenarios.yaml")
CONFIGURATION = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _spec(scenario="S1", distribution="baseline_distribution"):
    _, specs = resolve_scenarios(PHASE2_CONFIG, scenarios=[scenario], distributions=[distribution])
    return specs[0]


class IncrementSignTests(unittest.TestCase):
    def test_variant_increments_have_opposite_signs(self):
        a = ScoreTrackerConfig(missed_increment_mode="likelihood_penalty")
        b = ScoreTrackerConfig(missed_increment_mode="documented")
        self.assertLess(a.missed_increment, 0.0)
        self.assertGreater(b.missed_increment, 0.0)
        self.assertAlmostEqual(a.missed_increment, -b.missed_increment)

    def test_documented_formula_matches_paper_eq30(self):
        b = ScoreTrackerConfig(missed_increment_mode="documented", p_false_alarm=0.02, p_detect=0.90)
        expected = math.log((1.0 - 0.02) / (1.0 - 0.90))
        self.assertAlmostEqual(b.missed_increment, expected)
        self.assertGreater(expected, 0.0)  # positive because P_D > P_FA

    def test_variant_a_formula_is_the_negation(self):
        a = ScoreTrackerConfig(missed_increment_mode="likelihood_penalty", p_false_alarm=0.02, p_detect=0.90)
        self.assertAlmostEqual(a.missed_increment, math.log((1.0 - 0.90) / (1.0 - 0.02)))

    def test_birth_increment_is_positive_prior(self):
        config = ScoreTrackerConfig(p_birth=0.05, p_false_alarm=0.02)
        self.assertAlmostEqual(config.birth_increment, math.log(0.05 / 0.02))


class ParityTests(unittest.TestCase):
    def test_variant_nodes_differ_only_in_sign(self):
        shared = CONFIGURATION["shared_tracker_config"]
        node_a = _variant_node(shared, CONFIGURATION["variants"]["A_missed_decay"])
        node_b = _variant_node(shared, CONFIGURATION["variants"]["B_documented"])
        self.assertEqual(node_a["score"]["missed_increment_mode"], "likelihood_penalty")
        self.assertEqual(node_b["score"]["missed_increment_mode"], "documented")
        stripped_a = deepcopy(node_a)
        stripped_b = deepcopy(node_b)
        stripped_a["score"].pop("missed_increment_mode")
        stripped_b["score"].pop("missed_increment_mode")
        self.assertEqual(stripped_a, stripped_b)

    def test_generated_inputs_are_identical_for_repeated_generation(self):
        spec = _spec("S6", "paper_assumption_v1")
        first = _input_hashes(generate_scenario(spec))
        second = _input_hashes(generate_scenario(spec))
        self.assertEqual(first, second)

    def test_runner_enforces_parity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = run_matrix(
                CONFIG_PATH,
                root / "gen",
                root / "plots",
                root / "manifest.json",
                scenarios=["S0", "S6"],
                distributions=["baseline_distribution", "paper_assumption_v1"],
            )
            self.assertEqual(manifest["case_count"], 8)
            self.assertTrue(manifest["parity"]["ok"], manifest["parity"]["mismatches"])
            self.assertLess(manifest["computed_missed_increment_signs"]["A_missed_decay"], 0.0)
            self.assertGreater(manifest["computed_missed_increment_signs"]["B_documented"], 0.0)

    def test_default_output_namespace_is_phase3_1(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.output_root, Path("data/results/phase3_1"))
        self.assertEqual(args.plot_root, Path("outputs/figures/phase3_1"))
        self.assertEqual(args.manifest, Path("data/manifests/phase3_1_manifest.json"))


class OfflineAuditDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_classify_associations_and_candidates(self):
        truth = np.full((1, 1, 4), np.nan)
        truth[0, 0, :2] = [0.0, 0.0]
        detections = [[Detection(np.array([0.0, 0.0]), 0), Detection(np.array([50.0, 50.0]), None)]]
        simulation = SimulationData(
            truth, detections, np.array([0]), np.array([1]), ScenarioConfig(steps=1, num_targets=1)
        )
        estimates = [[TrackEstimate(1, np.array([0.0, 0.0, 0.0, 0.0]), True, 1, 0)]]
        result = TrackingResult(estimates, [[(1, 0)]])
        audit = association_outcome_diagnostics(simulation, result)
        self.assertEqual(audit["total_measurements"], 2.0)
        self.assertEqual(audit["associations_to_true"], 1.0)
        self.assertEqual(audit["associations_to_clutter"], 0.0)
        self.assertEqual(audit["unmatched_clutter"], 1.0)
        self.assertEqual(audit["candidates_born_from_clutter"], 1.0)
        self.assertEqual(audit["unmatched_true_detections"], 0.0)

    def test_clutter_only_sequence_is_suppressed_by_variant_a(self):
        rng = np.random.default_rng(11)
        frames = [[Detection(rng.uniform(0, 100, 2)) for _ in range(4)] for _ in range(30)]
        tracker_a = DynamicScoreTracker(ScoreTrackerConfig(missed_increment_mode="likelihood_penalty"))
        tracker_a.run(frames)
        tracker_b = DynamicScoreTracker(ScoreTrackerConfig(missed_increment_mode="documented"))
        tracker_b.run(frames)
        self.assertEqual(tracker_a.diagnostics["max_concurrent_confirmed"], 0.0)
        self.assertGreaterEqual(
            tracker_b.diagnostics["max_concurrent_confirmed"],
            tracker_a.diagnostics["max_concurrent_confirmed"],
        )


class IsolationAndDeterminismTests(unittest.TestCase):
    def test_label_mutation_does_not_change_tracker_output(self):
        node = _variant_node(CONFIGURATION["shared_tracker_config"], CONFIGURATION["variants"]["B_documented"])
        spec = _spec("S1", "baseline_distribution")
        simulation = generate_scenario(spec)
        labelled = [[Detection(d.position.copy(), d.source_target_id) for d in frame] for frame in simulation.detections]
        unlabelled = [[Detection(d.position.copy()) for d in frame] for frame in simulation.detections]

        def signature(frames):
            result = DynamicScoreTracker(_dynamic_config(node)).run(frames)
            return [[(e.track_id, tuple(np.round(e.state, 6))) for e in frame] for frame in result.estimates_by_frame]

        self.assertEqual(signature(labelled), signature(unlabelled))

    def test_repeatable_deterministic_fields(self):
        node = _variant_node(CONFIGURATION["shared_tracker_config"], CONFIGURATION["variants"]["A_missed_decay"])
        spec = _spec("S2", "baseline_distribution")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = _run_case("A", node, spec, 12.0, root / "g1", root / "p1")
            second = _run_case("A", node, spec, 12.0, root / "g2", root / "p2")
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["audit"], second["audit"])
        self.assertEqual(first["input_hashes"], second["input_hashes"])
        for case in (first, second):
            self.assertGreaterEqual(case["runtime_seconds"], 0.0)
            self.assertTrue(math.isfinite(case["runtime_seconds"]))
            self.assertTrue(all(math.isfinite(v) for v in case["metrics"].values()))


if __name__ == "__main__":
    unittest.main()
