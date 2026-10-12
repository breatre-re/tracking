"""Phase 3.1 score-direction audit runner.

Runs the unchanged Phase 2 S0-S7 matrix (both distributions) for two dynamic
score variants that differ *only* in the missed-detection increment direction,
and writes every artifact under a new ``phase3_1`` namespace. No Phase 0-3 file
is modified: this runner only imports existing Phase 2/3 code.

Parity is enforced and recorded: for each (distribution, scenario) the generated
inputs (truth + position measurements) are hashed and the resolved scenario
config, seed, tracker config (apart from the sign mode) and evaluator settings
must be identical across variants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import yaml

from .phase2_baseline_evaluation import plot_scenario
from .phase3_dynamic_score_evaluation import _dynamic_config
from src.evaluation.phase2_metrics import evaluate_scenario
from src.evaluation.phase3_1_audit import association_outcome_diagnostics
from src.evaluation.phase3_metrics import recovery_metrics
from src.simulation.dataset import Detection, SimulationData
from src.simulation.scenarios import ScenarioSpec, generate_scenario, resolve_scenarios
from src.tracking.dynamic_score_tracker import DynamicScoreTracker


SIGN_FIELDS = ("false_positives", "false_negatives", "id_switches", "fragmentation", "position_rmse", "mota_like")


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
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_hash(arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return digest.hexdigest()


def _input_hashes(simulation: SimulationData) -> Dict[str, str]:
    positions = [
        np.asarray([detection.position for detection in frame], dtype=float).reshape((-1, 2))
        for frame in simulation.detections
    ]
    labels = [
        np.asarray(
            [-1.0 if detection.source_target_id is None else float(detection.source_target_id) for detection in frame],
            dtype=float,
        )
        for frame in simulation.detections
    ]
    return {
        "truth_hash": _payload_hash([simulation.true_states]),
        "measurement_hash": _payload_hash(positions),
        "label_hash": _payload_hash(labels) if labels else _payload_hash([np.zeros(0)]),
    }


def _tracker_detections(simulation: SimulationData) -> List[List[Detection]]:
    return [
        [Detection(np.asarray(detection.position, dtype=float).copy()) for detection in frame]
        for frame in simulation.detections
    ]


def _variant_node(shared: dict, variant: dict) -> dict:
    node = deepcopy(shared)
    node.setdefault("score", {})
    node["score"]["missed_increment_mode"] = variant["missed_increment_mode"]
    return node


def _run_case(variant_name: str, node: dict, spec: ScenarioSpec, match_gate: float, output_root: Path, plot_root: Path) -> dict:
    simulation = generate_scenario(spec)
    hashes = _input_hashes(simulation)
    detections = _tracker_detections(simulation)

    tracker = DynamicScoreTracker(_dynamic_config(node))
    start = time.perf_counter()
    result = tracker.run(detections)
    runtime = time.perf_counter() - start

    metrics = evaluate_scenario(simulation, result, match_gate)
    recovery = recovery_metrics(simulation, result, spec.disappearance, match_gate)
    audit = association_outcome_diagnostics(simulation, result)

    case_dir = output_root / variant_name / spec.distribution / spec.name
    case_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = case_dir / f"{spec.name}_metrics.json"
    plot_path = plot_root / variant_name / spec.distribution / f"{spec.name}_trajectory.png"

    payload = {
        "variant": variant_name,
        "scenario": spec.name,
        "distribution": spec.distribution,
        "resolved_config": spec.resolved(),
        "match_gate": match_gate,
        "input_hashes": hashes,
        "tracker_config": node,
        "metrics": metrics,
        "recovery": recovery,
        "audit": audit,
        "diagnostics": deepcopy(getattr(tracker, "diagnostics", {})),
        "runtime_seconds": runtime,
        "notes": [
            "Tracker input contains positions only; source labels are used solely by the offline evaluator/audit.",
            "Only missed_increment_mode differs between variants A and B.",
            "runtime_seconds is wall-clock and not a parity field.",
        ],
    }
    metrics_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_scenario(simulation, result, spec, plot_path)

    return {
        "variant": variant_name,
        "distribution": spec.distribution,
        "scenario": spec.name,
        "resolved_config": spec.resolved(),
        "match_gate": match_gate,
        "input_hashes": hashes,
        "tracker_config": node,
        "metrics": metrics,
        "recovery": recovery,
        "audit": audit,
        "diagnostics": payload["diagnostics"],
        "runtime_seconds": runtime,
        "metrics_file": metrics_path.as_posix(),
        "metrics_sha256": _sha256(metrics_path),
        "plot_file": plot_path.as_posix(),
    }


def run_matrix(
    config_path: Path,
    output_root: Path,
    plot_root: Path,
    manifest_path: Path,
    scenarios: Optional[Sequence[str]] = None,
    distributions: Optional[Sequence[str]] = None,
    variants_filter: Optional[Sequence[str]] = None,
) -> dict:
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    shared = configuration["shared_tracker_config"]
    variants = configuration["variants"]
    selected_variants = list(variants_filter) if variants_filter else list(variants)
    unknown = [name for name in selected_variants if name not in variants]
    if unknown:
        raise ValueError(f"unknown variant(s): {', '.join(unknown)}")
    match_gate, specs = resolve_scenarios(
        Path(configuration.get("phase2_config", "configs/phase2_scenarios.yaml")),
        scenarios=scenarios,
        distributions=distributions,
    )

    cases: List[dict] = []
    variant_nodes = {}
    for variant_name in selected_variants:
        node = _variant_node(shared, variants[variant_name])
        variant_nodes[variant_name] = node
        for spec in specs:
            cases.append(_run_case(variant_name, node, spec, match_gate, output_root, plot_root))

    # Parity: identical inputs / resolved config / evaluator settings across variants.
    by_case: Dict[tuple, dict] = {}
    mismatches: List[str] = []
    for case in cases:
        key = (case["distribution"], case["scenario"])
        signature = {
            "input_hashes": case["input_hashes"],
            "resolved_config": case["resolved_config"],
            "match_gate": case["match_gate"],
            "tracker_config_without_sign": {
                k: v for k, v in case["tracker_config"].items() if k != "score"
            },
            "score_without_sign": {
                k: v for k, v in case["tracker_config"].get("score", {}).items() if k != "missed_increment_mode"
            },
        }
        if key not in by_case:
            by_case[key] = signature
        elif by_case[key] != signature:
            mismatches.append(f"{case['distribution']}/{case['scenario']}")

    audit = configuration.get("audit", {})
    p_detect = float(audit.get("p_detect", shared["score"]["p_detect"]))
    p_false_alarm = float(audit.get("p_false_alarm", shared["score"]["p_false_alarm"]))
    sign_values = {
        "A_missed_decay": float(np.log((1.0 - p_detect) / (1.0 - p_false_alarm))),
        "B_documented": float(np.log((1.0 - p_false_alarm) / (1.0 - p_detect))),
    }

    manifest = {
        "schema_version": 1,
        "config_file": config_path.as_posix(),
        "phase2_config_file": configuration.get("phase2_config", "configs/phase2_scenarios.yaml"),
        "match_gate": match_gate,
        "variant_labels": {name: variants[name].get("label", name) for name in selected_variants},
        "variant_nodes": variant_nodes,
        "computed_missed_increment_signs": sign_values,
        "audit": audit,
        "parity": {"ok": not mismatches, "mismatches": mismatches, "cases": len(by_case)},
        "case_count": len(cases),
        "cases": cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _print_summary(manifest: dict) -> None:
    print("missed-increment sign values:", manifest["computed_missed_increment_signs"])
    print("parity ok:", manifest["parity"]["ok"], "mismatches:", manifest["parity"]["mismatches"])
    header = (
        f"{'variant':<18}{'distribution':<22}{'scenario':<5}{'match':>7}{'FP':>7}{'FN':>7}{'IDS':>6}"
        f"{'frag':>6}{'RMSE':>8}{'MOTA':>8}{'recov':>7}{'clut_assoc':>11}{'clut_birth':>11}"
    )
    print(header)
    print("-" * len(header))
    for case in manifest["cases"]:
        m = case["metrics"]
        a = case["audit"]
        rate = case["recovery"]["recovery_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.2f}"
        print(
            f"{case['variant']:<18}{case['distribution']:<22}{case['scenario']:<5}"
            f"{m['matched_detections']:>7.0f}{m['false_positives']:>7.0f}{m['false_negatives']:>7.0f}"
            f"{m['id_switches']:>6.0f}{m['fragmentation']:>6.0f}{m['position_rmse']:>8.3f}"
            f"{m['mota_like']:>8.3f}{rate_text:>7}"
            f"{a['associations_to_clutter']:>11.0f}{a['candidates_born_from_clutter']:>11.0f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase3_1_score_audit.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/results/phase3_1"))
    parser.add_argument("--plot-root", type=Path, default=Path("outputs/figures/phase3_1"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/phase3_1_manifest.json"))
    parser.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario subset, e.g. S0,S6")
    parser.add_argument("--distributions", type=str, default="", help="Comma-separated distribution subset")
    parser.add_argument("--variants", type=str, default="", help="Comma-separated variant subset")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = [value for value in args.scenarios.split(",") if value] or None
    distributions = [value for value in args.distributions.split(",") if value] or None
    variants = [value for value in args.variants.split(",") if value] or None
    manifest = run_matrix(
        args.config,
        args.output_root,
        args.plot_root,
        args.manifest,
        scenarios=scenarios,
        distributions=distributions,
        variants_filter=variants,
    )
    if not args.quiet:
        _print_summary(manifest)
    print(f"\nmanifest: {args.manifest}")


if __name__ == "__main__":
    main()
