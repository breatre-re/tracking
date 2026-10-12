"""Phase 4 tracker comparison runner.

Runs the *unchanged* Phase 2 S0-S7 matrix (both distributions) against three
trackers and writes everything under the additive ``phase4`` namespace:

* ``gated_greedy``          - Phase 1/2 baseline association.
* ``dynamic_score_B``       - Phase 3 dynamic-score tracker, paper-faithful
                              documented missed increment (variant B).
* ``ctda_dynamic_score``    - CTDA association + BLUE filter + B dynamic score.

No Phase 0-3.1 file is modified; this runner only imports existing code.
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

from phase2_runner import plot_scenario
from phase3_runner import _dynamic_config
from src.evaluation.phase2_metrics import evaluate_scenario
from src.evaluation.phase3_1_audit import association_outcome_diagnostics
from src.evaluation.phase3_metrics import recovery_metrics
from src.models.ctda.adapter import CTDAAssociation
from src.models.ctda.tracker import CTDADynamicScoreTracker
from src.simulation.dataset import Detection, SimulationData
from src.simulation.scenarios import ScenarioSpec, generate_scenario, resolve_scenarios
from src.tracking.dynamic_score_tracker import DynamicScoreTracker
from tracking_prototype import GatedGreedyAssociation, MultiTargetTracker


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


def _input_hash(simulation: SimulationData) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(simulation.true_states, dtype=np.float64).tobytes())
    for frame in simulation.detections:
        positions = np.asarray([detection.position for detection in frame], dtype=float).reshape((-1, 2))
        digest.update(np.ascontiguousarray(positions, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _tracker_detections(simulation: SimulationData) -> List[List[Detection]]:
    return [
        [Detection(np.asarray(detection.position, dtype=float).copy()) for detection in frame]
        for frame in simulation.detections
    ]


def build_tracker(tracker_name: str, configuration: dict, checkpoint: Optional[Path], device: str):
    score_node = configuration["score_tracker"]
    if tracker_name == "gated_greedy":
        return MultiTargetTracker(GatedGreedyAssociation(), gate=9.0, min_hits=2, max_missed=5)
    if tracker_name == "dynamic_score_B":
        return DynamicScoreTracker(_dynamic_config(score_node))
    if tracker_name == "ctda_dynamic_score":
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(f"CTDA checkpoint not found: {checkpoint}")
        association = CTDAAssociation.from_checkpoint(
            Path(checkpoint),
            device=device,
            threshold=float(configuration["inference"]["association_threshold"]),
        )
        return CTDADynamicScoreTracker(_dynamic_config(score_node), association)
    raise ValueError(f"unknown tracker {tracker_name!r}")


def run_case(tracker_name: str, configuration: dict, checkpoint: Optional[Path], device: str, spec: ScenarioSpec, match_gate: float, output_root: Path, plot_root: Path) -> dict:
    simulation = generate_scenario(spec)
    detections = _tracker_detections(simulation)
    tracker = build_tracker(tracker_name, configuration, checkpoint, device)
    start = time.perf_counter()
    result = tracker.run(detections)
    runtime = time.perf_counter() - start

    metrics = evaluate_scenario(simulation, result, match_gate)
    recovery = recovery_metrics(simulation, result, spec.disappearance, match_gate)
    audit = association_outcome_diagnostics(simulation, result)

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
        "input_hash": _input_hash(simulation),
        "metrics": metrics,
        "recovery": recovery,
        "audit": audit,
        "diagnostics": deepcopy(getattr(tracker, "diagnostics", {})),
        "runtime_seconds": runtime,
        "notes": [
            "Tracker input contains positions only; ground-truth ids never reach the tracker.",
            "Phase 2 scenarios/evaluator and the Phase 3 B score direction are unchanged.",
            "runtime_seconds is wall-clock and not part of deterministic equality.",
        ],
    }
    metrics_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_scenario(simulation, result, spec, plot_path)

    return {
        "tracker": tracker_name,
        "distribution": spec.distribution,
        "scenario": spec.name,
        "resolved_config": spec.resolved(),
        "input_hash": payload["input_hash"],
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
    checkpoint: Optional[Path],
    output_root: Path,
    plot_root: Path,
    manifest_path: Path,
    scenarios: Optional[Sequence[str]] = None,
    distributions: Optional[Sequence[str]] = None,
    trackers: Optional[Sequence[str]] = None,
) -> dict:
    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = "cuda" if _cuda_available() else "cpu"
    match_gate, specs = resolve_scenarios(
        Path(configuration.get("phase2_config", "configs/phase2_scenarios.yaml")),
        scenarios=scenarios,
        distributions=distributions,
    )
    selected = list(trackers) if trackers else ["gated_greedy", "dynamic_score_B", "ctda_dynamic_score"]

    cases: List[dict] = []
    for tracker_name in selected:
        for spec in specs:
            cases.append(run_case(tracker_name, configuration, checkpoint, device, spec, match_gate, output_root, plot_root))

    # Parity: gated-greedy and dynamic_score_B must share identical inputs per case.
    hashes: Dict[tuple, str] = {}
    mismatches: List[str] = []
    for case in cases:
        key = (case["distribution"], case["scenario"])
        if key not in hashes:
            hashes[key] = case["input_hash"]
        elif hashes[key] != case["input_hash"]:
            mismatches.append(f"{case['distribution']}/{case['scenario']}")

    manifest = {
        "schema_version": 1,
        "config_file": config_path.as_posix(),
        "phase2_config_file": configuration.get("phase2_config", "configs/phase2_scenarios.yaml"),
        "match_gate": match_gate,
        "device": device,
        "trackers": selected,
        "checkpoint": None if checkpoint is None else Path(checkpoint).as_posix(),
        "checkpoint_sha256": _sha256(Path(checkpoint)) if checkpoint and Path(checkpoint).is_file() else None,
        "association_threshold": float(configuration["inference"]["association_threshold"]),
        "score_missed_increment_mode": configuration["score_tracker"]["score"]["missed_increment_mode"],
        "parity": {"ok": not mismatches, "mismatches": mismatches, "cases": len(hashes)},
        "case_count": len(cases),
        "cases": cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def _print_summary(manifest: dict) -> None:
    print("parity ok:", manifest["parity"]["ok"], "mismatches:", manifest["parity"]["mismatches"])
    header = (
        f"{'tracker':<20}{'distribution':<22}{'scenario':<5}{'match':>7}{'FP':>7}{'FN':>7}{'IDS':>6}"
        f"{'frag':>6}{'RMSE':>8}{'MOTA':>8}{'recov':>7}{'rt[s]':>9}"
    )
    print(header)
    print("-" * len(header))
    for case in manifest["cases"]:
        m = case["metrics"]
        rate = case["recovery"]["recovery_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.2f}"
        print(
            f"{case['tracker']:<20}{case['distribution']:<22}{case['scenario']:<5}"
            f"{m['matched_detections']:>7.0f}{m['false_positives']:>7.0f}{m['false_negatives']:>7.0f}"
            f"{m['id_switches']:>6.0f}{m['fragmentation']:>6.0f}{m['position_rmse']:>8.3f}"
            f"{m['mota_like']:>8.3f}{rate_text:>7}{case['runtime_seconds']:>9.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_ctda.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/phase4/row_column_3ch/checkpoint.pt"))
    parser.add_argument("--output-root", type=Path, default=Path("data/generated/phase4/tracker"))
    parser.add_argument("--plot-root", type=Path, default=Path("reports/phase4_plots"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/phase4_manifest.json"))
    parser.add_argument("--scenarios", type=str, default="")
    parser.add_argument("--distributions", type=str, default="")
    parser.add_argument("--trackers", type=str, default="")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = [value for value in args.scenarios.split(",") if value] or None
    distributions = [value for value in args.distributions.split(",") if value] or None
    trackers = [value for value in args.trackers.split(",") if value] or None
    manifest = run_matrix(
        args.config,
        args.checkpoint,
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
