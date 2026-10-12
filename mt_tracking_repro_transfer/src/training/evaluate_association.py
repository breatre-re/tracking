"""Phase 4 association-only evaluation on the held-out test split.

Loads a trained checkpoint and evaluates association quality on
``paper_assumption_v1/test`` (never used for training or checkpoint selection).
Reports the paper's association accuracy (Eq. 35) plus precision/recall/F1 after
threshold ``lambda_a`` + Kuhn-Munkres decoding, no-match handling counts, runtime
and a per-density (measurement-count) breakdown.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.ctda.features import PackedAssociationDataset
from src.models.ctda.model import CTDANet, CTDAConfig
from src.models.ctda.objective import association_metrics


def _empty_totals() -> dict:
    return {
        "cells": 0.0,
        "true_positives": 0.0,
        "false_positives": 0.0,
        "false_negatives": 0.0,
        "true_negatives": 0.0,
        "rows": 0.0,
        "cols": 0.0,
        "unmatched_rows": 0.0,
        "unmatched_cols": 0.0,
    }


def _finalize(totals: dict) -> dict:
    tp, fp, fn, tn = (totals["true_positives"], totals["false_positives"], totals["false_negatives"], totals["true_negatives"])
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        **totals,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
    }


def evaluate(checkpoint_path: Path, config: dict, split: str, device: str, max_examples: int | None = None) -> dict:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    configuration = payload.get("config", config)
    model = CTDANet(CTDAConfig.from_dict(configuration))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    features_config = configuration.get("features", config["features"])
    dataset_config = configuration.get("dataset", config["dataset"])
    inference = configuration.get("inference", config["inference"])
    threshold = float(inference.get("association_threshold", 0.1))
    one_to_one = bool(inference.get("one_to_one", True))

    dataset = PackedAssociationDataset(
        Path(split),
        pad_size=int(dataset_config["pad_size"]),
        channels=tuple(features_config["channels"]),
        x_bias=float(features_config["x_bias"]),
        pad_value=float(features_config["pad_value"]),
    )
    indices = range(len(dataset)) if max_examples is None else range(min(max_examples, len(dataset)))

    totals = _empty_totals()
    buckets = {"1-10": _empty_totals(), "11-20": _empty_totals(), "21-30": _empty_totals(), "31-40": _empty_totals()}
    per_example_accuracy = []
    started = time.perf_counter()
    with torch.no_grad():
        for index in indices:
            example = dataset.build_example(*dataset.index[index])
            tensor = torch.from_numpy(example.features[None]).to(device)
            row_tensor = torch.from_numpy(example.row_mask[None]).to(device)
            col_tensor = torch.from_numpy(example.col_mask[None]).to(device)
            logits = model(tensor, row_tensor, col_tensor)
            probabilities = torch.sigmoid(logits)[0].cpu().numpy()
            metrics = association_metrics(
                probabilities, example.target, example.row_mask, example.col_mask, threshold, one_to_one
            )
            for key in totals:
                totals[key] += metrics[key]
            bucket = buckets["1-10"] if metrics["cols"] <= 10 else buckets["11-20"] if metrics["cols"] <= 20 else buckets["21-30"] if metrics["cols"] <= 30 else buckets["31-40"]
            for key in bucket:
                bucket[key] += metrics[key]
            per_example_accuracy.append(metrics["accuracy"])
    elapsed = time.perf_counter() - started

    return {
        "checkpoint": checkpoint_path.as_posix(),
        "variant": payload.get("variant", configuration.get("model", {}).get("variant", "unknown")),
        "split": split,
        "examples": len(per_example_accuracy),
        "threshold": threshold,
        "one_to_one": one_to_one,
        "device": device,
        "runtime_seconds": elapsed,
        "aggregate": _finalize(totals),
        "mean_example_accuracy": float(np.mean(per_example_accuracy)) if per_example_accuracy else 0.0,
        "per_measurement_count": {name: _finalize(value) for name, value in buckets.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_ctda.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-examples", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split = args.split or Path(configuration["dataset"]["test_split"])
    device = args.device if torch.cuda.is_available() else "cpu"
    result = evaluate(
        args.checkpoint,
        configuration,
        str(split),
        device,
        max_examples=args.max_examples or None,
    )
    print(json.dumps({k: result[k] for k in ("variant", "examples", "threshold", "runtime_seconds")}, indent=2))
    print(json.dumps(result["aggregate"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
