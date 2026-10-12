"""Phase 2 runner: evaluate the gated-greedy baseline on a fixed scenario matrix.

It regenerates or loads labelled scenario data, strips the labels before the
tracker ever sees them, measures wall-clock runtime, computes the Phase 2 metric
set, and writes a per-case metrics JSON plus a trajectory plot. It does not
implement CTDA and does not change tracker association or lifecycle behaviour.

Usage:

    conda run -n mttrack-repro python -m src.experiments.phase2_baseline_evaluation \
        --config configs/phase2_scenarios.yaml \
        --output-root data/results/phase2 \
        --plot-root outputs/figures/phase2 \
        --manifest data/manifests/phase2_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.evaluation.phase2_metrics import evaluate_scenario
from src.simulation.dataset import Detection, SimulationData
from src.simulation.scenarios import ScenarioSpec, generate_scenario, resolve_scenarios
from src.tracking.baseline_tracker import GatedGreedyAssociation, MultiTargetTracker, TrackingResult


TRACKER_DEFAULTS = {"association": "gated-greedy", "gate": 9.0, "min_hits": 2, "max_missed": 5}


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
    """Strip source labels: the tracker only ever receives positions."""

    return [
        [Detection(np.asarray(detection.position, dtype=float).copy()) for detection in frame]
        for frame in simulation.detections
    ]


def run_tracker(
    simulation: SimulationData,
) -> tuple[TrackingResult, float]:
    tracker = MultiTargetTracker(
        GatedGreedyAssociation(),
        gate=TRACKER_DEFAULTS["gate"],
        min_hits=TRACKER_DEFAULTS["min_hits"],
        max_missed=TRACKER_DEFAULTS["max_missed"],
    )
    detections = _tracker_detections(simulation)
    start = time.perf_counter()
    result = tracker.run(detections)
    runtime = time.perf_counter() - start
    return result, runtime


def plot_scenario(simulation: SimulationData, result: TrackingResult, spec: ScenarioSpec, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(8, 8), dpi=100)

    measurements = np.asarray(
        [detection.position for frame in simulation.detections for detection in frame],
        dtype=float,
    ).reshape((-1, 2))
    if len(measurements):
        axes.scatter(measurements[:, 0], measurements[:, 1], s=4, c="0.78", label="measurements/clutter", zorder=1)

    truth = simulation.true_states
    for target in range(truth.shape[1]):
        xs, ys = truth[:, target, 0], truth[:, target, 1]
        mask = np.isfinite(xs)
        if mask.any():
            axes.plot(xs[mask], ys[mask], "-", lw=1.3, alpha=0.85, zorder=2)

    tracks: Dict[int, List[tuple]] = {}
    for frame, estimates in enumerate(result.estimates_by_frame):
        for estimate in estimates:
            tracks.setdefault(estimate.track_id, []).append((frame, estimate.state[0], estimate.state[1]))
    colormap = plt.get_cmap("tab20")
    for order, (track_id, points) in enumerate(sorted(tracks.items())):
        points.sort()
        axes.plot(
            [point[1] for point in points],
            [point[2] for point in points],
            "-",
            lw=1.4,
            color=colormap(order % 20),
            zorder=3,
        )

    axes.set_title(f"{spec.distribution} / {spec.name}: {spec.description}")
    axes.set_xlabel("x [m]")
    axes.set_ylabel("y [m]")
    axes.set_aspect("equal", adjustable="box")
    axes.grid(True, alpha=0.25)
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(handles, labels, loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def run_case(spec: ScenarioSpec, match_gate: float, output_root: Path, plot_root: Path) -> dict:
    simulation = generate_scenario(spec)
    result, runtime = run_tracker(simulation)
    metrics = evaluate_scenario(simulation, result, match_gate)

    case_dir = output_root / spec.distribution / spec.name
    case_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = case_dir / f"{spec.name}_metrics.json"
    plot_path = plot_root / spec.distribution / f"{spec.name}_trajectory.png"

    payload = {
        "scenario": spec.name,
        "distribution": spec.distribution,
        "resolved_config": spec.resolved(),
        "match_gate": match_gate,
        "tracker": dict(TRACKER_DEFAULTS),
        "metrics": metrics,
        "runtime_seconds": runtime,
        "notes": [
            "Tracker input contains positions only; source labels are used solely by the evaluator.",
            "runtime_seconds is wall-clock and not part of the deterministic metrics.",
        ],
    }
    metrics_path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_scenario(simulation, result, spec, plot_path)

    return {
        "distribution": spec.distribution,
        "scenario": spec.name,
        "resolved_config": spec.resolved(),
        "metrics": metrics,
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
) -> dict:
    match_gate, specs = resolve_scenarios(config_path, scenarios=scenarios, distributions=distributions)
    cases = [run_case(spec, match_gate, output_root, plot_root) for spec in specs]

    manifest = {
        "schema_version": 1,
        "config_file": config_path.as_posix(),
        "match_gate": match_gate,
        "tracker": dict(TRACKER_DEFAULTS),
        "case_count": len(cases),
        "cases": cases,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _print_summary(manifest: dict) -> None:
    header = (
        f"{'distribution':<22}{'scenario':<6}{'match':>7}{'FP':>7}{'FN':>7}"
        f"{'IDS':>6}{'frag':>6}{'RMSE':>9}{'MOTA':>8}{'runtime[s]':>12}"
    )
    print(header)
    print("-" * len(header))
    for case in manifest["cases"]:
        metrics = case["metrics"]
        print(
            f"{case['distribution']:<22}{case['scenario']:<6}"
            f"{metrics['matched_detections']:>7.0f}{metrics['false_positives']:>7.0f}"
            f"{metrics['false_negatives']:>7.0f}{metrics['id_switches']:>6.0f}"
            f"{metrics['fragmentation']:>6.0f}{metrics['position_rmse']:>9.3f}"
            f"{metrics['mota_like']:>8.3f}{case['runtime_seconds']:>12.4f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_scenarios.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("data/results/phase2"))
    parser.add_argument("--plot-root", type=Path, default=Path("outputs/figures/phase2"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/phase2_manifest.json"))
    parser.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario subset, e.g. S0,S1")
    parser.add_argument(
        "--distributions",
        type=str,
        default="",
        help="Comma-separated distribution subset, e.g. baseline_distribution",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print the summary table")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenarios = [value for value in args.scenarios.split(",") if value] or None
    distributions = [value for value in args.distributions.split(",") if value] or None
    manifest = run_matrix(
        args.config,
        args.output_root,
        args.plot_root,
        args.manifest,
        scenarios=scenarios,
        distributions=distributions,
    )
    if not args.quiet:
        _print_summary(manifest)
    print(f"\nmanifest: {args.manifest}")


if __name__ == "__main__":
    main()
