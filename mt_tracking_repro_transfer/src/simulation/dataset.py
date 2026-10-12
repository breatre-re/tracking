"""Reproducible synthetic 2D tracking data generation and storage."""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class ScenarioConfig:
    """Parameters for the synthetic 2D scenario."""

    steps: int = 80
    num_targets: int = 6
    area_size: float = 100.0
    detection_probability: float = 0.90
    measurement_std: float = 1.5
    clutter_rate: float = 1.5
    process_accel_std: float = 0.08
    seed: int = 7


@dataclass
class Detection:
    position: Array
    source_target_id: Optional[int] = None


@dataclass
class SimulationData:
    true_states: Array
    detections: List[List[Detection]]
    birth_times: Array
    death_times: Array
    config: ScenarioConfig


@dataclass
class DatasetSample:
    simulation: SimulationData
    measurement_truth_ids: List[Array]
    clutter_events: Array
    missed_detection_events: Array


def _validate_config(config: ScenarioConfig) -> None:
    if config.num_targets < 1 or config.steps < 2:
        raise ValueError("num_targets must be >= 1 and steps must be >= 2")
    if config.area_size <= 10:
        raise ValueError("area_size must be > 10")
    if not 0.0 <= config.detection_probability <= 1.0:
        raise ValueError("detection_probability must be between 0 and 1")
    if config.measurement_std < 0 or config.process_accel_std < 0:
        raise ValueError("measurement_std and process_accel_std must be >= 0")
    if config.clutter_rate < 0:
        raise ValueError("clutter_rate must be >= 0")


def _make_lifetimes(config: ScenarioConfig) -> Tuple[Array, Array]:
    """Create deterministic births/deaths, including temporary life overlap."""

    n = config.num_targets
    births = np.zeros(n, dtype=int)
    deaths = np.full(n, config.steps, dtype=int)
    if n >= 1:
        births[0], deaths[0] = 0, config.steps
    if n >= 2:
        births[1], deaths[1] = 0, config.steps
    if n >= 3:
        births[2], deaths[2] = min(10, config.steps - 1), config.steps
    if n >= 4:
        births[3], deaths[3] = 0, min(config.steps, 62)
    for i in range(4, n):
        births[i] = min(5 + 4 * (i - 4), config.steps - 1)
        deaths[i] = min(config.steps, births[i] + 42 + 3 * (i % 3))
    return births, deaths


def _initial_motion_parameters(
    config: ScenarioConfig, rng: np.random.Generator
) -> Tuple[Array, Array]:
    """Return initial positions/velocities; first four form a crossing scene."""

    n = config.num_targets
    positions = rng.uniform(15.0, config.area_size - 15.0, size=(n, 2))
    velocities = rng.uniform(-0.8, 0.8, size=(n, 2))
    if n >= 1:
        positions[0], velocities[0] = [20.0, 20.0], [0.95, 0.60]
    if n >= 2:
        positions[1], velocities[1] = [78.0, 58.0], [-0.95, -0.60]
    if n >= 3:
        positions[2], velocities[2] = [20.0, 78.0], [0.65, -0.55]
    if n >= 4:
        positions[3], velocities[3] = [82.0, 20.0], [-0.55, 0.65]
    return positions, velocities


def simulate_scenario(config: ScenarioConfig) -> SimulationData:
    """Generate truth and detections with misses, noise and Poisson clutter."""

    _validate_config(config)
    rng = np.random.default_rng(config.seed)
    births, deaths = _make_lifetimes(config)
    positions, velocities = _initial_motion_parameters(config, rng)
    truth = np.full((config.steps, config.num_targets, 4), np.nan, dtype=float)
    detections: List[List[Detection]] = []

    for time in range(config.steps):
        frame_detections: List[Detection] = []
        for target_id in range(config.num_targets):
            active = births[target_id] <= time < deaths[target_id]
            if active:
                if time > births[target_id]:
                    velocities[target_id] += rng.normal(
                        0.0, config.process_accel_std, 2
                    )
                    positions[target_id] += velocities[target_id]
                    for axis in (0, 1):
                        if (
                            positions[target_id, axis] < 5.0
                            or positions[target_id, axis] > config.area_size - 5.0
                        ):
                            velocities[target_id, axis] *= -1.0
                            positions[target_id, axis] = np.clip(
                                positions[target_id, axis],
                                5.0,
                                config.area_size - 5.0,
                            )
                truth[time, target_id, :2] = positions[target_id]
                truth[time, target_id, 2:] = velocities[target_id]
                if rng.random() < config.detection_probability:
                    measured = positions[target_id] + rng.normal(
                        0.0, config.measurement_std, 2
                    )
                    frame_detections.append(Detection(measured, target_id))

        clutter_count = int(rng.poisson(config.clutter_rate))
        for _ in range(clutter_count):
            frame_detections.append(
                Detection(rng.uniform(0.0, config.area_size, 2), None)
            )
        rng.shuffle(frame_detections)
        detections.append(frame_detections)

    return SimulationData(truth, detections, births, deaths, config)


def sample_with_events(config: ScenarioConfig) -> DatasetSample:
    """Generate one sample and materialize frame-indexed source/event labels."""

    simulation = simulate_scenario(config)
    measurement_truth_ids: List[Array] = []
    missed_events: List[Tuple[int, int]] = []
    clutter_events: List[Tuple[int, int]] = []

    for frame, detections in enumerate(simulation.detections):
        frame_ids = np.array(
            [-1 if d.source_target_id is None else d.source_target_id for d in detections],
            dtype=np.int64,
        )
        measurement_truth_ids.append(frame_ids)
        clutter_events.extend(
            (frame, index)
            for index, truth_id in enumerate(frame_ids)
            if truth_id == -1
        )
        active_targets = np.flatnonzero(np.isfinite(simulation.true_states[frame, :, 0]))
        detected_targets = set(int(value) for value in frame_ids if value >= 0)
        missed_events.extend(
            (frame, int(target_id))
            for target_id in active_targets
            if int(target_id) not in detected_targets
        )

    return DatasetSample(
        simulation=simulation,
        measurement_truth_ids=measurement_truth_ids,
        clutter_events=np.asarray(clutter_events, dtype=np.int64).reshape((-1, 2)),
        missed_detection_events=np.asarray(missed_events, dtype=np.int64).reshape((-1, 2)),
    )


def _write_deterministic_npz(path: Path, arrays: dict[str, Array]) -> None:
    """Write an NPZ with fixed ZIP metadata for byte-identical reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)


def write_dataset_split(
    path: Path,
    *,
    split: str,
    config: ScenarioConfig,
    sample_count: int,
    seed_start: int,
    target_count_range: Optional[Tuple[int, int]] = None,
) -> dict[str, object]:
    """Generate and save a split; sample seed equals seed_start + sample index."""

    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    target_min, target_max = target_count_range or (
        config.num_targets,
        config.num_targets,
    )
    if target_min < 1 or target_max < target_min:
        raise ValueError("target_count_range must satisfy 1 <= min <= max")
    sample_seeds = np.arange(seed_start, seed_start + sample_count, dtype=np.int64)
    target_counts = np.empty(sample_count, dtype=np.int64)
    sample_truth: List[Array] = []
    measurements: List[Array] = []
    labels: List[Array] = []
    offsets = np.zeros((sample_count, config.steps + 1), dtype=np.int64)
    birth_event_arrays: List[Array] = []
    death_event_arrays: List[Array] = []
    missed_event_arrays: List[Array] = []
    clutter_event_arrays: List[Array] = []
    clutter_counts = np.zeros((sample_count, config.steps), dtype=np.int64)
    measurement_cursor = 0

    for sample_index, seed in enumerate(sample_seeds):
        target_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 0x54415247])
        )
        num_targets = int(target_rng.integers(target_min, target_max + 1))
        target_counts[sample_index] = num_targets
        sample_config = ScenarioConfig(
            **{**asdict(config), "num_targets": num_targets, "seed": int(seed)}
        )
        sample = sample_with_events(sample_config)
        simulation = sample.simulation
        padded_truth = np.full(
            (config.steps, target_max, 4), np.nan, dtype=np.float64
        )
        padded_truth[:, :num_targets] = simulation.true_states
        sample_truth.append(padded_truth)
        target_ids = np.arange(num_targets, dtype=np.int64)
        sample_prefix = np.full(num_targets, sample_index, dtype=np.int64)
        birth_event_arrays.append(
            np.column_stack((sample_prefix, target_ids, simulation.birth_times))
        )
        death_event_arrays.append(
            np.column_stack((sample_prefix, target_ids, simulation.death_times))
        )

        offsets[sample_index, 0] = measurement_cursor
        for frame, detections in enumerate(simulation.detections):
            frame_measurements = np.asarray(
                [d.position for d in detections], dtype=np.float64
            ).reshape((-1, 2))
            frame_labels = sample.measurement_truth_ids[frame]
            measurements.append(frame_measurements)
            labels.append(frame_labels)
            clutter_counts[sample_index, frame] = int(
                np.count_nonzero(frame_labels == -1)
            )
            measurement_cursor += len(frame_measurements)
            offsets[sample_index, frame + 1] = measurement_cursor
        if len(sample.missed_detection_events):
            missed_event_arrays.append(
                np.column_stack(
                    (
                        np.full(
                            len(sample.missed_detection_events),
                            sample_index,
                            dtype=np.int64,
                        ),
                        sample.missed_detection_events,
                    )
                )
            )
        if len(sample.clutter_events):
            clutter_event_arrays.append(
                np.column_stack(
                    (
                        np.full(
                            len(sample.clutter_events), sample_index, dtype=np.int64
                        ),
                        sample.clutter_events,
                    )
                )
            )

    all_measurements = (
        np.concatenate(measurements, axis=0)
        if measurements
        else np.empty((0, 2), dtype=np.float64)
    )
    all_labels = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    empty_events = np.empty((0, 3), dtype=np.int64)
    birth_events = np.concatenate(birth_event_arrays, axis=0)
    death_events = np.concatenate(death_event_arrays, axis=0)
    missed_events = (
        np.concatenate(missed_event_arrays, axis=0)
        if missed_event_arrays
        else empty_events.copy()
    )
    clutter_events = (
        np.concatenate(clutter_event_arrays, axis=0)
        if clutter_event_arrays
        else empty_events.copy()
    )
    stored_config = asdict(config)
    stored_config.update(
        {
            "split": split,
            "sample_count": sample_count,
            "sample_seed_policy": "sample_seed = seed_start + sample_index",
            "seed_start": seed_start,
            "target_count_range": [target_min, target_max],
            "target_count_seed_stream": "SeedSequence([sample_seed, 0x54415247])",
        }
    )
    arrays = {
        "schema_version": np.asarray("2", dtype=np.str_),
        "split": np.asarray(split, dtype=np.str_),
        "config_json": np.asarray(
            json.dumps(stored_config, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "sample_seeds": sample_seeds,
        "target_counts": target_counts,
        "truth_states": np.stack(sample_truth, axis=0),
        "measurements": all_measurements,
        "measurement_truth_ids": all_labels,
        "frame_offsets": offsets,
        "birth_events": birth_events,
        "death_events": death_events,
        "missed_detection_events": missed_events,
        "clutter_events": clutter_events,
        "clutter_counts_per_sample_frame": clutter_counts,
    }
    _write_deterministic_npz(path, arrays)

    return {
        "split": split,
        "sample_count": sample_count,
        "seed_start": seed_start,
        "seed_end": int(sample_seeds[-1]),
        "total_measurements": int(len(all_measurements)),
        "target_count_range": [target_min, target_max],
        "target_count_mean": float(np.mean(target_counts)),
        "target_count_histogram": {
            str(int(value)): int(np.count_nonzero(target_counts == value))
            for value in np.unique(target_counts)
        },
        "clutter_count_per_frame_mean": float(np.mean(clutter_counts)),
        "clutter_count_per_frame_std": float(np.std(clutter_counts)),
        "truth_states_shape": list(arrays["truth_states"].shape),
        "measurement_shape": list(all_measurements.shape),
        "event_counts": {
            "birth": len(birth_events),
            "death": len(death_events),
            "missed_detection": len(missed_events),
            "clutter": len(clutter_events),
        },
    }


def load_dataset_sample(path: Path, sample_index: int = 0) -> DatasetSample:
    """Load one sample from a packed split without exposing labels to tracking."""

    with np.load(path, allow_pickle=False) as dataset:
        truth_states = dataset["truth_states"]
        sample_count, steps, _, _ = truth_states.shape
        if not 0 <= sample_index < sample_count:
            raise IndexError(
                f"sample_index {sample_index} out of range for {sample_count} samples"
            )
        metadata = json.loads(str(dataset["config_json"].item()))
        seed = int(dataset["sample_seeds"][sample_index])
        # schema 1 splits (the original Phase 1 datasets) predate per-sample
        # target counts and pad nothing, so the target dimension is fixed.
        if "target_counts" in dataset.files:
            num_targets = int(dataset["target_counts"][sample_index])
        else:
            num_targets = int(truth_states.shape[2])
        config = ScenarioConfig(
            steps=steps,
            num_targets=num_targets,
            area_size=float(metadata["area_size"]),
            detection_probability=float(metadata["detection_probability"]),
            measurement_std=float(metadata["measurement_std"]),
            clutter_rate=float(metadata["clutter_rate"]),
            process_accel_std=float(metadata["process_accel_std"]),
            seed=seed,
        )
        offsets = dataset["frame_offsets"][sample_index]
        all_measurements = dataset["measurements"]
        all_labels = dataset["measurement_truth_ids"]
        frame_detections: List[List[Detection]] = []
        frame_labels: List[Array] = []
        for frame in range(steps):
            start, end = int(offsets[frame]), int(offsets[frame + 1])
            labels = all_labels[start:end].copy()
            frame_labels.append(labels)
            frame_detections.append(
                [
                    Detection(position.copy())
                    for position in all_measurements[start:end]
                ]
            )

        birth_times = np.full(num_targets, steps, dtype=np.int64)
        death_times = np.full(num_targets, steps, dtype=np.int64)
        for split_name in ("birth_events", "death_events"):
            for event_sample, target_id, frame in dataset[split_name]:
                if int(event_sample) != sample_index:
                    continue
                if split_name == "birth_events":
                    birth_times[int(target_id)] = int(frame)
                else:
                    death_times[int(target_id)] = int(frame)
        missed = dataset["missed_detection_events"]
        clutter = dataset["clutter_events"]
        missed_sample_events = missed[missed[:, 0] == sample_index, 1:3].copy()
        clutter_sample_events = clutter[clutter[:, 0] == sample_index, 1:3].copy()

    return DatasetSample(
        simulation=SimulationData(
            true_states=truth_states[sample_index, :, :num_targets].copy(),
            detections=frame_detections,
            birth_times=birth_times,
            death_times=death_times,
            config=config,
        ),
        measurement_truth_ids=frame_labels,
        clutter_events=clutter_sample_events,
        missed_detection_events=missed_sample_events,
    )
