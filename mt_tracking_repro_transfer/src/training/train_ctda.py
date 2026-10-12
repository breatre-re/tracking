"""Phase 4 CTDA training entrypoint.

Trains one association network on ``paper_assumption_v1/train`` only, with the
shared settings from the plan (8 epochs, batch 256, Adam, lr 1e-4). Checkpoint
selection uses a train-internal held-out *sequence* split so test sequences never
influence model selection. Writes checkpoint, resolved config, epoch metrics and
input hashes under ``models/checkpoints/phase4/runs/<variant>/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.models.ctda.features import PackedAssociationDataset, split_sample_indices
from src.models.ctda.model import CTDANet, CTDAConfig
from src.models.ctda.objective import focal_loss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _make_dataset(configuration: dict, sample_indices: List[int], max_frames: int | None) -> PackedAssociationDataset:
    dataset = configuration["dataset"]
    features = configuration["features"]
    return PackedAssociationDataset(
        Path(dataset["train_split"]),
        pad_size=int(dataset["pad_size"]),
        channels=tuple(features["channels"]),
        x_bias=float(features["x_bias"]),
        pad_value=float(features["pad_value"]),
        sample_indices=sample_indices,
        max_frames=max_frames,
    )


def train(configuration: dict, variant: str, output_dir: Path, limits: Dict[str, int | None]) -> dict:
    dataset_config = configuration["dataset"]
    training_config = configuration["training"]
    seed = int(training_config.get("seed", 20261012))
    _set_seeds(seed, bool(training_config.get("deterministic", True)))
    device = torch.device(training_config.get("device", "cpu") if torch.cuda.is_available() else "cpu")

    with np.load(Path(dataset_config["train_split"]), allow_pickle=False) as handle:
        sample_count = int(handle["frame_offsets"].shape[0])
    if limits.get("samples"):
        sample_count = min(sample_count, int(limits["samples"]))
    training_samples, validation_samples = split_sample_indices(
        sample_count, float(dataset_config.get("validation_fraction", 0.1)), seed
    )

    max_frames = int(limits["frames"]) if limits.get("frames") else None
    training_dataset = _make_dataset(configuration, training_samples, max_frames)
    validation_dataset = _make_dataset(configuration, validation_samples, max_frames)

    loader = DataLoader(
        training_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config.get("num_workers", 0)),
    )

    model_configuration = dict(configuration)
    model_configuration["model"] = {**configuration["model"], "variant": variant}
    model = CTDANet(CTDAConfig.from_dict(model_configuration)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["lr"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    alpha = float(training_config.get("focal_alpha", 0.75))
    gamma = float(training_config.get("focal_gamma", 2.0))

    epochs = int(limits["epochs"] or training_config["epochs"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []
    best_validation = float("inf")
    best_epoch = -1
    started = time.perf_counter()

    def run_epoch(loader_: DataLoader, train_mode: bool) -> float:
        model.train(train_mode)
        total, count = 0.0, 0
        for batch in loader_:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            row_mask = batch["row_mask"].to(device)
            col_mask = batch["col_mask"].to(device)
            logits = model(features, row_mask, col_mask)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"non-finite logits in {'training' if train_mode else 'validation'} batch; "
                    "check attention masks and input features"
                )
            loss = focal_loss(logits, target, row_mask, col_mask, alpha, gamma)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite focal loss in {'training' if train_mode else 'validation'} batch"
                )
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if any(
                    parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("non-finite gradient during CTDA training")
                optimizer.step()
            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    for epoch in range(epochs):
        train_loss = run_epoch(loader, True)
        with torch.no_grad():
            validation_loss = run_epoch(validation_loader, False)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss, "seconds": None}
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": model_configuration,
                    "variant": variant,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                output_dir / "checkpoint.pt",
            )
        print(f"[{variant}] epoch {epoch}: train={train_loss:.6f} val={validation_loss:.6f}")

    elapsed = time.perf_counter() - started
    for entry in history:
        entry["seconds"] = elapsed / max(len(history), 1)

    summary = {
        "variant": variant,
        "device": str(device),
        "epochs": epochs,
        "batch_size": int(training_config["batch_size"]),
        "optimizer": "adam",
        "lr": float(training_config["lr"]),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation,
        "train_examples": len(training_dataset),
        "validation_examples": len(validation_dataset),
        "history": history,
        "training_seconds": elapsed,
        "train_split": dataset_config["train_split"],
        "train_split_sha256": _sha256(Path(dataset_config["train_split"])),
        "validation_samples": validation_samples[:32],
        "deterministic": bool(training_config.get("deterministic", True)),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase4_ctda.yaml"))
    parser.add_argument("--variant", default="", help="standard_transformer | row_column | row_column_3ch | all")
    parser.add_argument("--output-root", type=Path, default=Path("models/checkpoints/phase4/runs"))
    parser.add_argument("--epochs", type=int, default=0, help="Override epoch count (smoke runs)")
    parser.add_argument("--limit-train-samples", type=int, default=0)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override DataLoader workers (default: config)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        configuration["training"]["device"] = args.device
    if args.num_workers >= 0:
        configuration["training"]["num_workers"] = args.num_workers
    variants = (
        ["standard_transformer", "row_column", "row_column_3ch"]
        if args.variant in ("", "all")
        else [args.variant]
    )
    limits = {
        "epochs": args.epochs or None,
        "samples": args.limit_train_samples or None,
        "frames": args.limit_frames or None,
    }
    for variant in variants:
        summary = train(configuration, variant, args.output_root / variant, limits)
        print(json.dumps({k: summary[k] for k in ("variant", "epochs", "best_epoch", "best_validation_loss")}, indent=2))


if __name__ == "__main__":
    main()
