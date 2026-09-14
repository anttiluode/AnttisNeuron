"""Stable linear branch dynamics and local nonlinearities."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def spectral_radius(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square")
    return float(np.max(np.abs(np.linalg.eigvals(a))))


def activation(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if name == "identity":
        return x
    if name == "square":
        return x * x
    if name == "nmda":
        positive = np.maximum(x, 0.0)
        return positive * positive / (1.0 + positive)
    raise ValueError(f"unknown activation: {name}")


def decay_ratio(mu_slow: float, mu_fast: float, t: int | float) -> float:
    """Relative continuous-time slow/fast amplitude ratio for equal initial amplitudes."""
    if not (0.0 <= mu_slow < mu_fast):
        raise ValueError("require 0 <= mu_slow < mu_fast")
    if t < 0:
        raise ValueError("t must be non-negative")
    return float(np.exp((mu_fast - mu_slow) * float(t)))


@dataclass
class LinearBranch:
    """A minimal discrete-time stable branch: d <- A d + b u, r = c^T d."""

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        self.c = np.asarray(self.c, dtype=float)
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")
        n = self.A.shape[0]
        if self.b.shape != (n,) or self.c.shape != (n,):
            raise ValueError("incompatible branch dimensions")
        if not all(np.all(np.isfinite(z)) for z in (self.A, self.b, self.c)):
            raise ValueError("branch parameters must be finite")
        if spectral_radius(self.A) >= 1.0:
            raise ValueError("branch must be discrete-time stable")
        self.state = np.zeros(n, dtype=float)

    def reset(self) -> None:
        self.state.fill(0.0)

    def step(self, drive: float) -> float:
        self.state = self.A @ self.state + self.b * float(drive)
        return float(self.c @ self.state)
