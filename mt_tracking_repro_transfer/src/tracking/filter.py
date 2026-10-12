"""Recursive linear-Gaussian filter for the Phase 3 dynamic-score tracker.

This is an explicit engineering simplification. The paper's 2D benchmark uses a
5D coordinated-turn state with range/bearing measurements and BLUE/EKF updates
(see ``docs/plan/IMPLEMENTATION_SPEC.md`` Sections 2-5). Phase 2's scenes provide Cartesian
``[x, y]`` positions and must stay unchanged, so Phase 3 keeps the Cartesian
measurement semantics and implements a 4D ``[x, y, vx, vy]`` constant-velocity
Kalman filter as the recursive BLUE filter.

The filter exposes the prediction, the innovation and its covariance, the
Mahalanobis distance, the Gaussian innovation log-likelihood and an update. The
covariance update uses the Joseph form plus an explicit symmetrization so the
covariance stays positive semi-definite under repeated updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Array = np.ndarray


@dataclass
class FilterConfig:
    """Filter noise / initialization parameters (all reproduction assumptions)."""

    process_accel_std: float = 0.10
    measurement_noise_std: float = 1.5
    initial_position_var: float = 4.0
    initial_velocity_var: float = 9.0


def _transition(dt: float) -> Array:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _process_covariance(config: FilterConfig, dt: float) -> Array:
    q = float(config.process_accel_std) ** 2
    block = q * np.array(
        [[dt ** 4 / 4.0, dt ** 3 / 2.0], [dt ** 3 / 2.0, dt ** 2]],
        dtype=float,
    )
    covariance = np.zeros((4, 4), dtype=float)
    # State ordering is [x, y, vx, vy]: (x, vx) live at indices [0, 2].
    covariance[np.ix_([0, 2], [0, 2])] = block
    covariance[np.ix_([1, 3], [1, 3])] = block
    return covariance


class CartesianKalmanFilter:
    """4D constant-velocity Kalman filter over Cartesian positions.

    Implements the :class:`Filter` contract consumed by the dynamic-score
    tracker: ``predict`` advances the state, ``innovation`` / ``mahalanobis`` /
    ``log_likelihood`` score an incoming measurement, and ``update`` performs the
    recursive BLUE/Kalman correction.
    """

    def __init__(self, position: Array, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()
        position = np.asarray(position, dtype=float)
        self.state = np.array([position[0], position[1], 0.0, 0.0], dtype=float)
        self.covariance = np.diag(
            [
                float(self.config.initial_position_var),
                float(self.config.initial_position_var),
                float(self.config.initial_velocity_var),
                float(self.config.initial_velocity_var),
            ]
        )
        self.measurement_noise = np.eye(2) * float(self.config.measurement_noise_std) ** 2

    @property
    def _observation(self) -> Array:
        return np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)

    @property
    def position(self) -> Array:
        return self.state[[0, 1]].copy()

    def predict(self, dt: float = 1.0) -> None:
        transition = _transition(dt)
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T
            + _process_covariance(self.config, dt)
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def innovation(self, measurement: Array) -> Tuple[Array, Array]:
        observation = self._observation
        residual = np.asarray(measurement, dtype=float) - observation @ self.state
        covariance = observation @ self.covariance @ observation.T + self.measurement_noise
        covariance = 0.5 * (covariance + covariance.T)
        return residual, covariance

    def mahalanobis(self, measurement: Array) -> float:
        residual, covariance = self.innovation(measurement)
        return float(residual @ np.linalg.solve(covariance, residual))

    def log_likelihood(self, measurement: Array) -> float:
        residual, covariance = self.innovation(measurement)
        dimension = residual.shape[0]
        # log N(residual; 0, S)
        sign, log_det = np.linalg.slogdet(covariance)
        if sign <= 0:  # pragma: no cover - defensive, covariance should be SPD
            return float("-inf")
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        return float(-0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic))

    def update(self, measurement: Array) -> float:
        """Recursive BLUE update; returns the innovation log-likelihood."""

        observation = self._observation
        residual, covariance = self.innovation(measurement)
        gain = np.linalg.solve(covariance, observation @ self.covariance).T
        self.state = self.state + gain @ residual
        identity = np.eye(4)
        # Joseph form keeps the covariance symmetric positive semi-definite.
        joseph = (identity - gain @ observation) @ self.covariance @ (identity - gain @ observation).T
        joseph = joseph + gain @ self.measurement_noise @ gain.T
        self.covariance = 0.5 * (joseph + joseph.T)

        dimension = residual.shape[0]
        sign, log_det = np.linalg.slogdet(covariance)
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        log_likelihood = float(-0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic))
        return log_likelihood
