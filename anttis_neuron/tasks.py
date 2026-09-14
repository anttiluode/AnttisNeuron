"""Deterministic synthetic tasks for spectral-neuron experiments."""
from __future__ import annotations

import numpy as np


def _orthonormal_modes(rng: np.random.Generator, dim: int = 6) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    return q[:, :2].T


def _sample_mode_energy_split(
    rng: np.random.Generator,
    n: int,
    modes: np.ndarray,
    *,
    high_std: float = 3.4,
    low_std: float = 0.25,
    background_std: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("n must be >= 2")
    dim = modes.shape[1]
    y = np.arange(n, dtype=int) % 2
    rng.shuffle(y)
    x = rng.normal(scale=background_std, size=(n, dim))

    coeff_background = x @ modes.T
    x = x - coeff_background @ modes
    coeff = np.empty((n, 2), dtype=float)
    for i, label in enumerate(y):
        stds = (high_std, low_std) if label == 0 else (low_std, high_std)
        coeff[i, 0] = rng.normal(scale=stds[0])
        coeff[i, 1] = rng.normal(scale=stds[1])
    x += coeff @ modes
    return x, y


def make_mode_energy_task(
    n_train: int,
    n_test: int,
    *,
    seed: int = 0,
    dim: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Variance-only binary task with known orthonormal discriminating modes."""
    if dim < 2:
        raise ValueError("dim must be >= 2")
    rng = np.random.default_rng(seed)
    modes = _orthonormal_modes(rng, dim=dim)
    x_train, y_train = _sample_mode_energy_split(rng, n_train, modes)
    x_test, y_test = _sample_mode_energy_split(rng, n_test, modes)
    return x_train, y_train, x_test, y_test, modes


def make_local_oja_streams(
    modes: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
    dominant_std: float = 3.0,
    background_std: float = 0.35,
) -> list[np.ndarray]:
    """Create one local stream per mode, each dominated by its designated direction."""
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 2:
        raise ValueError("modes must have shape (n_modes, dim)")
    rng = np.random.default_rng(seed)
    streams: list[np.ndarray] = []
    for mode in modes:
        base = rng.normal(scale=background_std, size=(n_samples, modes.shape[1]))
        coeff = rng.normal(scale=dominant_std, size=(n_samples, 1))
        streams.append(base + coeff * mode[None, :])
    return streams
