"""Oja-style principal-mode learning utilities."""
from __future__ import annotations

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return arr / norm


def absolute_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Return sign-invariant cosine alignment in [0, 1]."""
    return float(abs(np.dot(_normalize(a), _normalize(b))))


def principal_eigenvector(cov: np.ndarray) -> np.ndarray:
    """Return a normalized leading eigenvector of a symmetric covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be a square matrix")
    if not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    vals, vecs = np.linalg.eigh(cov)
    return _normalize(vecs[:, int(np.argmax(vals))])


def oja_fit(
    samples: np.ndarray,
    *,
    lr: float = 1e-3,
    epochs: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Fit one Oja unit to centered samples and return its normalized weight vector."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("samples must have shape (n_samples>=2, n_features>=1)")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite")
    if lr <= 0.0 or epochs < 1:
        raise ValueError("lr must be positive and epochs >= 1")

    centered = x - x.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    w = _normalize(rng.normal(size=x.shape[1]))

    for _ in range(epochs):
        for sample in centered:
            y = float(np.dot(w, sample))
            w += lr * y * (sample - y * w)
            norm = float(np.linalg.norm(w))
            if not np.isfinite(norm) or norm == 0.0:
                raise FloatingPointError("Oja update became non-finite")
            w /= norm
    return w
