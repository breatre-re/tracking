"""CLI for creating reproducible Phase 1 datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from .dataset import ScenarioConfig, write_dataset_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_from_config(
    config_path: Path,
    profile: str,
    output_root: Path,
    manifest_root: Path,
) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "config_file": config_path.as_posix(),
        "seed_policy": (
            "Each sample uses default_rng(seed_start + sample_index); target-count "
            "sampling, when configured, uses an independent SeedSequence stream."
        ),
        "distributions": {},
    }

    if "distributions" in configuration:
        distributions = configuration["distributions"]
        selected = list(distributions) if profile == "all" else [profile]
        unknown = [name for name in selected if name not in distributions]
        if unknown:
            raise ValueError(f"Unknown distribution profile(s): {', '.join(unknown)}")
        for distribution_name in selected:
            distribution = distributions[distribution_name]
            scenario = ScenarioConfig(**distribution["scenario"])
            result: dict[str, Any] = {
                "scenario": asdict(scenario),
                "assumptions": distribution.get("assumptions", []),
                "splits": {},
            }
            for split, settings in distribution["datasets"].items():
                destination = (
                    output_root
                    / distribution_name
                    / split
                    / f"{split}_dataset.npz"
                )
                target_range = settings.get(
                    "target_count_range",
                    distribution.get("target_count_range"),
                )
                split_info = write_dataset_split(
                    destination,
                    split=split,
                    config=scenario,
                    sample_count=int(settings["samples"]),
                    seed_start=int(settings["seed_start"]),
                    target_count_range=(
                        tuple(int(value) for value in target_range)
                        if target_range is not None
                        else None
                    ),
                )
                split_info["file"] = destination.as_posix()
                split_info["sha256"] = _sha256(destination)
                result["splits"][split] = split_info
            manifest["distributions"][distribution_name] = result
        manifest_name = "phase1_5_manifest.json"
    else:
        scenario = ScenarioConfig(**configuration["scenario"])
        profiles = configuration["datasets"]
        selected = list(profiles) if profile == "all" else [profile]
        unknown = [name for name in selected if name not in profiles]
        if unknown:
            raise ValueError(f"Unknown profile(s): {', '.join(unknown)}")

        manifest.update({"scenario": asdict(scenario), "splits": {}})
        for split in selected:
            settings = profiles[split]
            destination = output_root / split / f"{split}_dataset.npz"
            split_info = write_dataset_split(
                destination,
                split=split,
                config=scenario,
                sample_count=int(settings["samples"]),
                seed_start=int(settings["seed_start"]),
            )
            split_info["file"] = destination.as_posix()
            split_info["sha256"] = _sha256(destination)
            manifest["splits"][split] = split_info
        manifest_name = "phase1_manifest.json"

    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"manifest: {manifest_path}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_2d.yaml"))
    parser.add_argument("--profile", default="smoke")
    parser.add_argument("--output-root", type=Path, default=Path("data/datasets/phase1"))
    parser.add_argument("--manifest-root", type=Path, default=Path("data/manifests"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_from_config(
        args.config, args.profile, args.output_root, args.manifest_root
    )


if __name__ == "__main__":
    main()
