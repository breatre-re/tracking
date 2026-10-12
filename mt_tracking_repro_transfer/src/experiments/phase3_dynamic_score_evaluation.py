"""Phase 3 runner: compare the baseline and the dynamic-score tracker.

It runs the *unchanged* Phase 2 S0-S7 matrix (both distributions) against both
tracker variants, reusing the Phase 2 scenario generator and evaluator, and adds
the Phase 3 recovery / lifecycle diagnostics. Results are written under a new
``phase3`` namespace; no Phase 1/1.5/2 artifact is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

from .phase2_baseline_evaluation import plot_scenario
from src.evaluation.phase2_metrics import evaluate_scenario
from src.evaluation.phase3_metrics import recovery_metrics
from src.simulation.dataset import Detection, SimulationData
from src.simulation.scenarios import ScenarioSpec, generate_scenario, resolve_scenarios
from src.tracking.dynamic_score_tracker import DynamicScoreTracker, ScoreTrackerConfig
from src.tracking.baseline_tracker import GatedGreedyAssociation, MultiTargetTracker, TrackingResult


PHASE2_CONFIG = Path("configs/phase2_scenarios.yaml")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracker_detections(simulation: SimulationData) -> List[List[Detection]]:
    return [
        [Detection(np.asarray(detection.position, dtype=float).copy()) for detection in frame]
        for frame in simulation.detections
    ]


def _dynamic_config(node: dict) -> ScoreTrackerConfig:
    filter_node = node.get("filter", {})
    gate_node = node.get("gate", {})
    score_node = node.get("score", {})
    lifecycle_node = node.get("lifecycle", {})
    return ScoreTrackerConfig(
        process_accel_std=float(filter_node.get("process_accel_std", 0.10)),
        measurement_noise_std=float(filter_node.get("measurement_noise_std", 1.5)),
        initial_position_var=float(filter_node.get("initial_position_var", 4.0)),
        initial_velocity_var=float(filter_node.get("initial_velocity_var", 9.0)),
        mahalanobis_gate=float(gate_node.get("mahalanobis_gate", 9.21)),
        physical_gate=float(gate_node.get("physical_gate", 10.0)),
        lost_gate_scale=float(gate_node.get("lost_gate_scale", 2.5)),
        score_decay=float(score_node.get("decay", 0.95)),
        score_history_lag=int(score_node.get("history_lag", 5)),
        score_confirm=float(score_node.get("confirm_threshold", 3.0)),
        p_birth=float(score_node.get("p_birth", 0.05)),
        p_false_alarm=float(score_node.get("p_false_alarm", 0.02)),
        p_detect=float(score_node.get("p_detect", 0.90)),
        max_missed=int(lifecycle_node.get("max_missed", 5)),
        candidate_expiry=int(lifecycle_node.get("candidate_expiry", 3)),
        lost_expiry=int(lifecycle_node.get("lost_expiry", 15)),
        reappearance_reward=float(lifecycle_node.get("reappearance_reward", 0.0)),
        report_lost=bool(lifecycle_node.get("report_lost", True)),
        missed_increment_mode=str(score_node.get("missed_increment_mode", "documented")),
    )


def _make_tracker(tracker_name: str, node: dict):
    kind = node.get("kind", tracker_name)
    if kind == "gated_greedy":
        return MultiTargetTracker(
            GatedGreedyAssociation(),
            gate=float(node.get("gate", 9.0)),
            min_hits=int(node.get("min_hits", 2)),
            max_missed=int(node.get("max_missed", 5)),
        )
    if kind == "dynamic_score":
        return DynamicScoreTracker(_dynamic_config(node))
    raise ValueError(f"unknown tracker kind {kind!r}")


def run_tracker(simulation: SimulationData, tracker_name: str, node: dict) -> tuple:
    tracker = _make_tracker(tracker_name, node)
    detections = _tracker_detections(simulation)
    start = time.perf_counter()
    result = tracker.run(detections)
    runtime = time.perf_counter() - start
    diagnostics = deepcopy(getattr(tracker, "diagnostics", {}))
    return result, runtime, diagnostics


def run_case(tracker_name: str, node: dict, spec: ScenarioSpec, match_gate: float, output_root: Path, plot_root: Path) -> dict:
    simulation = generate_scenario(spec)
    result, runtime, diagnostics = run_tracker(simulation, tracker_name, node)

    metrics = evaluate_scenario(simulation, result, match_gate)
    recovery = recovery_metrics(simulation, result, spec.disappearance, match_gate)

    case_dir = output_root / tracker_name / spec.distribution / spec.name
    case_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = case_dir / f"{spec.name}_metrics.json"
    plot_path = plot_root / tracker_name / spec.distribution / f"{spec.name}_trajectory.png"

    payload = {
        "tracker": tracker_name,
        "scenario": spec.name,
        "distribution": spec.distribution,
        "resolved_config": spec.resolved(),
        "match_gate": match_gate,
        "tracker_config": node,
        "metrics": metrics,
        "recovery": recovery,
        "diagnostics": diagnostics,
        "runtime_seconds": runtime,
        "notes": [
            "Tracker input contains positions only; source labels are used solely by the evaluator.",
            "runtime_seconds is wall-clock and not part of the deterministic fields.",
        ],
    }
    metrics_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_scenario(simulation, result, spec, plot_path)

    return {
        "tracker": tracker_name,
        "distribution": spec.distribution,
        "scenario": spec.name,
        "resolved_config": spec.resolved(),
        "tracker_config": node,
        "metrics": metrics,
        "recovery": recovery,
        "diagnostics": diagnostics,
        "runtime_seconds": runtime,
        "metrics_file": metrics_path.as_posix(),
        "metrics_sha256": _sha256(metrics_path),
        "plot_file": plot_path.as_posix(),
    }


def run_matrix(
    tracker_config_path: Path,
    output_root: Path,
    plot_root: Path,
    manifest_path: Path,
    scenarios: Optional[Sequence[str]] = None,
    distributions: Optional[Sequence[str]] = None,
    trackers: Optional[Sequence[str]] = None,
) -> dict:
    tracker_configuration = yaml.safe_load(tracker_config_path.read_text(encoding="utf-8"))
    match_gate, specs = resolve_scenarios(PHASE2_CONFIG, scenarios=scenarios, distributions=distributions)
    selected_trackers = list(trackers) if trackers else list(tracker_configuration["trackers"])

    cases: List[dict] = []
    for tracker_name in selected_trackers:
        node = tracker_configuration["trackers"][tracker_name]
        for spec in specs:
            cases.append(run_case(tracker_name, node, spec, match_gate, output_root, plot_root))

    manifest = {
        "schema_version": 1,
        "phase2_config_file": PHASE2_CONFIG.as_posix(),
        "tracker_config_file": tracker_config_path.as_posix(),
        "match_gate": match_gate,
        "trackers": selected_trackers,
        "assumptions": tracker_configuration.get("assumptions", []),
        "case_count": len(cases),
        "cases": cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _print_summary(manifest: dict) -> None:
    header = (
        f"{'tracker':<15}{'distribution':<22}{'scenario':<10}{'match':>7}{'FP':>7}{'FN':>7}"
        f"{'IDS':>6}{'frag':>6}{'RMSE':>8}{'MOTA':>8}{'recov':>8}{'rt[s]':>9}"
    )
    print(header)
    print("-" * len(header))
    for case in manifest["cases"]:
        metrics = case["metrics"]
        recovery = case["recovery"]
        rate = recovery["recovery_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.2f}"
        print(
            f"{case['tracker']:<15}{case['distribution']:<22}{case['scenario']:<10}"
            f"{metrics['matched_detections']:>7.0f}{metrics['false_positives']:>7.0f}"
            f"{metrics['false_negatives']:>7.0f}{metrics['id_switches']:>6.0f}"
            f"{metrics['fragmentation']:>6.0f}{metrics['position_rmse']:>8.3f}"
            f"{metrics['mota_like']:>8.3f}{rate_text:>8}{case['runtime_seconds']:>9.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker-config", type=Path, default=Path("configs/phase3_dynamic_tracker.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/results/phase3"))
    parser.add_argument("--plot-root", type=Path, default=Path("outputs/figures/phase3"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/phase3_manifest.json"))
    parser.add_argument("--scenarios", type=str, default="")
    parser.add_argument("--distributions", type=str, default="")
    parser.add_argument("--trackers", type=str, default="", help="Comma-separated subset: baseline,dynamic_score")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = [value for value in args.scenarios.split(",") if value] or None
    distributions = [value for value in args.distributions.split(",") if value] or None
    trackers = [value for value in args.trackers.split(",") if value] or None
    manifest = run_matrix(
        args.tracker_config,
        args.output_root,
        args.plot_root,
        args.manifest,
        scenarios=scenarios,
        distributions=distributions,
        trackers=trackers,
    )
    if not args.quiet:
        _print_summary(manifest)
    print(f"\nmanifest: {args.manifest}")


if __name__ == "__main__":
    main()
