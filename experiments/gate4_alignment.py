"""Gate 4: can recurrent physical dynamics reshape statistics toward a physical eigenmode?"""
from __future__ import annotations

import numpy as np

from anttis_neuron.oja import absolute_cosine, oja_fit
from experiments.common import main


def _rotation(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def _state_observations(x: np.ndarray, a: np.ndarray, b: np.ndarray, burn: int = 500) -> np.ndarray:
    d = np.zeros(2, dtype=float)
    out = []
    for i, sample in enumerate(x):
        d = a @ d + b @ sample
        if i >= burn:
            out.append(b.T @ d)
    z = np.asarray(out)
    rms = float(np.sqrt(np.mean(np.sum(z * z, axis=1))))
    return z / rms


def _learn_alignment(samples: np.ndarray, physical_modes: np.ndarray, seed: int) -> float:
    learned = oja_fit(samples, lr=0.0008, epochs=3, seed=seed)
    return max(absolute_cosine(learned, mode) for mode in physical_modes)


def run(seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)

    theta = np.deg2rad(17.0)
    b = _rotation(theta)
    physical_modes = b.copy()

    external_angle = np.deg2rad(45.0 - 17.0)
    r_ext = _rotation(external_angle)
    cov = r_ext @ np.diag([2.0, 1.0]) @ r_ext.T
    x = rng.multivariate_normal(np.zeros(2), cov, size=26000)
    x_scale = float(np.sqrt(np.mean(np.sum(x * x, axis=1))))
    fixed_samples = x / x_scale

    a = np.diag([0.97, 0.65])
    coupled_samples = _state_observations(x, a, b)
    no_memory_samples = _state_observations(x, np.zeros((2, 2)), b)

    fixed = _learn_alignment(fixed_samples, physical_modes, seed + 1)
    coupled = _learn_alignment(coupled_samples, physical_modes, seed + 2)
    no_memory = _learn_alignment(no_memory_samples, physical_modes, seed + 3)

    return {
        "gate": 4,
        "seed": seed,
        "fixed_alignment": fixed,
        "coupled_alignment": coupled,
        "no_memory_alignment": no_memory,
        "delta_alignment": coupled - fixed,
        "slow_eigenvalue": float(a[0, 0]),
        "fast_eigenvalue": float(a[1, 1]),
        "interpretation": "existence test; positive delta is not assumed by CI",
    }


if __name__ == "__main__":
    main(run)
