"""Static spectral branch neuron identities and lightweight readouts."""
from __future__ import annotations

import numpy as np

from .branch import activation


def _as_2d_samples(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("x must have shape (n_samples, n_features)")
    return arr


def static_features(
    x: np.ndarray,
    filters: np.ndarray,
    nonlinearity: str = "identity",
) -> np.ndarray:
    x = _as_2d_samples(x)
    filters = np.asarray(filters, dtype=float)
    if filters.ndim != 2 or filters.shape[1] != x.shape[1]:
        raise ValueError("filters must have shape (n_branches, n_features)")
    return activation(nonlinearity, x @ filters.T)


def static_voltage(
    x: np.ndarray,
    filters: np.ndarray,
    output_weights: np.ndarray,
    *,
    bias: float = 0.0,
    nonlinearity: str = "identity",
) -> np.ndarray:
    phi = static_features(x, filters, nonlinearity)
    a = np.asarray(output_weights, dtype=float)
    if a.shape != (phi.shape[1],):
        raise ValueError("output_weights must match number of branches")
    return phi @ a + float(bias)


def collapsed_linear_filter(filters: np.ndarray, output_weights: np.ndarray) -> np.ndarray:
    m = np.asarray(filters, dtype=float)
    a = np.asarray(output_weights, dtype=float)
    if m.ndim != 2 or a.shape != (m.shape[0],):
        raise ValueError("incompatible filters/output_weights")
    return a @ m


def quadratic_matrix(filters: np.ndarray, output_weights: np.ndarray) -> np.ndarray:
    m = np.asarray(filters, dtype=float)
    a = np.asarray(output_weights, dtype=float)
    if m.ndim != 2 or a.shape != (m.shape[0],):
        raise ValueError("incompatible filters/output_weights")
    return np.einsum("j,ji,jk->ik", a, m, m)


def ridge_fit(features: np.ndarray, targets: np.ndarray, l2: float = 1e-6) -> tuple[np.ndarray, float]:
    """Fit a linear readout to binary targets encoded as 0/1, returning signed-score weights and bias."""
    x = _as_2d_samples(features)
    y = np.asarray(targets)
    if y.shape != (x.shape[0],):
        raise ValueError("targets must have shape (n_samples,)")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("targets must be binary 0/1")
    if l2 < 0.0:
        raise ValueError("l2 must be non-negative")
    y_signed = 2.0 * y.astype(float) - 1.0
    design = np.column_stack([x, np.ones(x.shape[0])])
    reg = np.eye(design.shape[1]) * float(l2)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ y_signed)
    return beta[:-1], float(beta[-1])


def accuracy_from_voltage(voltage: np.ndarray, targets: np.ndarray) -> float:
    v = np.asarray(voltage, dtype=float)
    y = np.asarray(targets)
    if v.shape != y.shape:
        raise ValueError("voltage and targets must have the same shape")
    pred = (v >= 0.0).astype(int)
    return float(np.mean(pred == y))
