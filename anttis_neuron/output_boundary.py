"""Minimal output-boundary primitives for Gate 7 load-compensation tests."""
from __future__ import annotations

import numpy as np

from .cable import weighted_laplacian


def driving_point_conductance(
    n_nodes: int,
    edges: np.ndarray,
    conductances: np.ndarray,
    *,
    leak: float,
    coupling: float,
    soma_node: int = 0,
) -> float:
    """Return steady-state driving-point conductance at one graph node.

    The continuous conductance matrix is ``G = leak*I + coupling*L(g)``.
    Eliminating every node except ``soma_node`` gives the scalar Schur
    complement seen by that node.
    """
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    if not 0 <= soma_node < n_nodes:
        raise ValueError("soma_node outside graph")
    if not np.isfinite(leak) or leak <= 0.0:
        raise ValueError("leak must be finite and positive")
    if not np.isfinite(coupling) or coupling < 0.0:
        raise ValueError("coupling must be finite and nonnegative")

    lap = weighted_laplacian(n_nodes, edges, conductances)
    matrix = leak * np.eye(n_nodes, dtype=float) + coupling * lap
    if n_nodes == 1:
        load = float(matrix[soma_node, soma_node])
    else:
        rest = np.asarray([i for i in range(n_nodes) if i != soma_node], dtype=int)
        g00 = float(matrix[soma_node, soma_node])
        g0r = matrix[soma_node, rest]
        grr = matrix[np.ix_(rest, rest)]
        gr0 = matrix[rest, soma_node]
        load = g00 - float(g0r @ np.linalg.solve(grr, gr0))
    if not np.isfinite(load) or load <= 0.0:
        raise FloatingPointError("driving-point conductance must be finite and positive")
    return load


def rate_boundary(
    voltage: np.ndarray,
    *,
    gain: float,
    theta: float,
    beta: float,
) -> np.ndarray:
    """Map soma voltage to a deterministic bounded firing-rate surrogate."""
    v = np.asarray(voltage, dtype=float)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("voltage must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(v)):
        raise ValueError("voltage must be finite")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if not np.isfinite(theta):
        raise ValueError("theta must be finite")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    logits = np.clip(beta * (gain * v - theta), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def calibrate_threshold(
    voltage: np.ndarray,
    *,
    gain: float,
    beta: float,
    target_rate: float,
    iterations: int = 80,
) -> float:
    """Choose threshold by deterministic bisection to match a mean rate."""
    v = np.asarray(voltage, dtype=float)
    if v.ndim != 1 or len(v) == 0 or not np.all(np.isfinite(v)):
        raise ValueError("voltage must be a finite nonempty vector")
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError("gain must be finite and positive")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    if not np.isfinite(target_rate) or not 0.0 < target_rate < 1.0:
        raise ValueError("target_rate must lie strictly between zero and one")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    driven = gain * v
    margin = 60.0 / beta
    low = float(np.min(driven) - margin)
    high = float(np.max(driven) + margin)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        mean_rate = float(np.mean(rate_boundary(v, gain=gain, theta=mid, beta=beta)))
        if mean_rate > target_rate:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def homeostatic_gain(
    gain: float,
    mean_rate: float,
    *,
    target_rate: float = 0.25,
    eta: float = 0.5,
    gain_min: float = 0.2,
    gain_max: float = 5.0,
) -> float:
    """Update output gain from local firing-rate error only."""
    values = (gain, mean_rate, target_rate, eta, gain_min, gain_max)
    if not all(np.isfinite(value) for value in values):
        raise ValueError("homeostatic values must be finite")
    if gain <= 0.0 or gain_min <= 0.0 or gain_max <= gain_min:
        raise ValueError("require positive gain bounds with gain_min < gain_max")
    if not 0.0 <= mean_rate <= 1.0 or not 0.0 <= target_rate <= 1.0:
        raise ValueError("rates must lie in [0, 1]")
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    log_gain = np.log(gain) + eta * (target_rate - mean_rate)
    clipped = np.clip(log_gain, np.log(gain_min), np.log(gain_max))
    return float(np.exp(clipped))


def transfer_metrics(
    scales: np.ndarray,
    rates: np.ndarray,
    reference_rates: np.ndarray,
    *,
    rheobase_rate: float = 0.10,
) -> dict:
    """Summarize one five-point input-output transfer curve."""
    x = np.asarray(scales, dtype=float)
    y = np.asarray(rates, dtype=float)
    ref = np.asarray(reference_rates, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or ref.shape != x.shape or len(x) < 3:
        raise ValueError("scales, rates, and reference_rates must be matching vectors")
    if not all(np.all(np.isfinite(value)) for value in (x, y, ref)):
        raise ValueError("transfer arrays must be finite")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError("scales must be strictly increasing")
    if np.any(y < 0.0) or np.any(y > 1.0) or np.any(ref < 0.0) or np.any(ref > 1.0):
        raise ValueError("rates must lie in [0, 1]")
    if not np.isfinite(rheobase_rate) or not 0.0 <= rheobase_rate <= 1.0:
        raise ValueError("rheobase_rate must lie in [0, 1]")

    rmse = float(np.sqrt(np.mean((y - ref) ** 2)))
    rheobase: float | None = None
    crossings = np.flatnonzero(y >= rheobase_rate)
    if len(crossings):
        index = int(crossings[0])
        if index == 0:
            rheobase = float(x[0])
        else:
            x0, x1 = float(x[index - 1]), float(x[index])
            y0, y1 = float(y[index - 1]), float(y[index])
            if abs(y1 - y0) <= 1e-15:
                rheobase = x1
            else:
                fraction = (rheobase_rate - y0) / (y1 - y0)
                rheobase = float(x0 + fraction * (x1 - x0))

    center = int(np.argmin(np.abs(x - 1.0)))
    if center == 0 or center == len(x) - 1:
        raise ValueError("unit scale must have neighbors for F-I gain")
    central_x = x[center - 1 : center + 2]
    central_y = y[center - 1 : center + 2]
    fi_gain = float(np.polyfit(central_x, central_y, 1)[0])

    return {
        "curve_rmse": rmse,
        "rheobase": rheobase,
        "fi_gain": fi_gain,
        "unit_rate": float(y[center]),
    }
