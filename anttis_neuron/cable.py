"""Branched graph cable dynamics and alignment-blind local structural adaptation."""
from __future__ import annotations

import numpy as np

from .branch import spectral_radius


def default_tree() -> tuple[int, np.ndarray]:
    """Return the fixed 11-node tree used by Gate 5."""
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [2, 4],
            [4, 5],
            [5, 6],
            [2, 7],
            [7, 8],
            [8, 9],
            [9, 10],
        ],
        dtype=int,
    )
    return 11, edges


def _validate_graph(n_nodes: int, edges: np.ndarray, conductances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    e = np.asarray(edges, dtype=int)
    g = np.asarray(conductances, dtype=float)
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    if g.shape != (len(e),):
        raise ValueError("conductances must have one value per edge")
    if np.any(e < 0) or np.any(e >= n_nodes):
        raise ValueError("edge index outside graph")
    if not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError("conductances must be finite and positive")
    return e, g


def weighted_laplacian(n_nodes: int, edges: np.ndarray, conductances: np.ndarray) -> np.ndarray:
    """Symmetric weighted graph Laplacian."""
    e, g = _validate_graph(n_nodes, edges, conductances)
    lap = np.zeros((n_nodes, n_nodes), dtype=float)
    for (i, j), weight in zip(e, g):
        i = int(i)
        j = int(j)
        lap[i, i] += weight
        lap[j, j] += weight
        lap[i, j] -= weight
        lap[j, i] -= weight
    return lap


def cable_operator(
    n_nodes: int,
    edges: np.ndarray,
    conductances: np.ndarray,
    *,
    dt: float,
    leak: float,
    coupling: float,
) -> np.ndarray:
    """Build a stable discrete leaky-diffusion cable operator."""
    if not all(np.isfinite(v) for v in (dt, leak, coupling)):
        raise ValueError("dt, leak, and coupling must be finite")
    if dt <= 0.0 or leak <= 0.0 or coupling < 0.0:
        raise ValueError("require dt > 0, leak > 0, coupling >= 0")
    lap = weighted_laplacian(n_nodes, edges, conductances)
    a = np.eye(n_nodes, dtype=float) - dt * (leak * np.eye(n_nodes) + coupling * lap)
    if spectral_radius(a) >= 1.0:
        raise ValueError("cable operator must be discrete-time stable")
    return a


def default_ports_and_sensors(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Four distal input ports and four bounded state sensors."""
    if n_nodes != 11:
        raise ValueError("default ports/sensors require the 11-node tree")
    ports = (3, 6, 9, 10)
    sensors = (1, 5, 8, 10)
    b = np.zeros((n_nodes, 4), dtype=float)
    s = np.zeros((4, n_nodes), dtype=float)
    for col, node in enumerate(ports):
        b[node, col] = 1.0
    for row, node in enumerate(sensors):
        s[row, node] = 1.0
    return b, s


def simulate_cable(
    inputs: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    *,
    burn: int = 0,
    return_states: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate zero-state cable dynamics and return bounded observations.

    Full states are optionally returned for strictly local edge-plasticity rules. They
    are not needed by the Oja learner, which sees only the bounded observations.
    """
    x = np.asarray(inputs, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    s = np.asarray(s, dtype=float)
    if x.ndim != 2:
        raise ValueError("inputs must have shape (time, input_dim)")
    n = a.shape[0]
    if a.shape != (n, n) or b.shape[0] != n or s.shape[1] != n or b.shape[1] != x.shape[1]:
        raise ValueError("incompatible cable dimensions")
    if burn < 0 or burn > len(x):
        raise ValueError("burn must be between 0 and input length")
    if not all(np.all(np.isfinite(v)) for v in (x, a, b, s)):
        raise ValueError("simulation arrays must be finite")
    state = np.zeros(n, dtype=float)
    observations: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for t, sample in enumerate(x):
        state = a @ state + b @ sample
        if t >= burn:
            observations.append(s @ state)
            if return_states:
                states.append(state.copy())
    if observations:
        z = np.asarray(observations, dtype=float)
    else:
        z = np.empty((0, s.shape[0]), dtype=float)
    if not return_states:
        return z
    if states:
        full = np.asarray(states, dtype=float)
    else:
        full = np.empty((0, n), dtype=float)
    return z, full


def visible_physical_modes(a: np.ndarray, s: np.ndarray, *, exclude_uniform: bool = True) -> np.ndarray:
    """Return unique normalized sensor-space projections of physical eigenmodes."""
    a = np.asarray(a, dtype=float)
    s = np.asarray(s, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("A must be square")
    n = a.shape[0]
    if s.ndim != 2 or s.shape[1] != n:
        raise ValueError("sensor matrix has incompatible shape")
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("visible_physical_modes requires a symmetric operator")

    values, vectors = np.linalg.eigh(a)
    order = np.argsort(np.abs(values))[::-1]
    uniform = np.ones(n, dtype=float) / np.sqrt(float(n))
    uniform_index = int(np.argmax(np.abs(vectors.T @ uniform))) if exclude_uniform else -1

    visible: list[np.ndarray] = []
    for index in order:
        if exclude_uniform and int(index) == uniform_index:
            continue
        projected = s @ vectors[:, index]
        norm = float(np.linalg.norm(projected))
        if norm <= 1e-12:
            continue
        mode = projected / norm
        if any(abs(float(np.dot(mode, prior))) > 1.0 - 1e-10 for prior in visible):
            continue
        visible.append(mode)

    if not visible:
        return np.empty((0, s.shape[0]), dtype=float)
    return np.asarray(visible, dtype=float)


def edge_difference_energy(states: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean squared voltage difference available locally at each edge."""
    d = np.asarray(states, dtype=float)
    e = np.asarray(edges, dtype=int)
    if d.ndim != 2:
        raise ValueError("states must have shape (time, n_nodes)")
    if len(d) == 0:
        raise ValueError("states cannot be empty")
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    if np.any(e < 0) or np.any(e >= d.shape[1]):
        raise ValueError("edge index outside state array")
    if not np.all(np.isfinite(d)):
        raise ValueError("states must be finite")
    diffs = d[:, e[:, 0]] - d[:, e[:, 1]]
    return np.mean(diffs * diffs, axis=0)


def normalize_mean_conductance(
    conductances: np.ndarray,
    target_mean: float,
    g_min: float,
    g_max: float,
) -> np.ndarray:
    """Clip positive conductances while preserving a requested feasible mean."""
    g = np.asarray(conductances, dtype=float).copy()
    if g.ndim != 1 or len(g) == 0 or not np.all(np.isfinite(g)):
        raise ValueError("conductances must be a finite nonempty vector")
    if not (0.0 < g_min <= target_mean <= g_max):
        raise ValueError("target mean must lie inside positive conductance bounds")
    if g_min >= g_max:
        raise ValueError("require g_min < g_max")
    g = np.clip(g, g_min, g_max)
    for _ in range(20):
        current = float(np.mean(g))
        if abs(current - target_mean) < 1e-12:
            break
        g = np.clip(g * (target_mean / current), g_min, g_max)
    if abs(float(np.mean(g)) - target_mean) >= 1e-10:
        raise FloatingPointError("mean conductance normalization did not converge")
    return g


def adapt_conductances(
    states: np.ndarray,
    edges: np.ndarray,
    conductances: np.ndarray,
    *,
    eta: float,
    q_target: float,
    g_min: float,
    g_max: float,
    mode: str = "local",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Slow local homeostatic update with no spectral or task-level supervision."""
    g = np.asarray(conductances, dtype=float)
    if g.ndim != 1 or len(g) != len(np.asarray(edges)):
        raise ValueError("conductances must have one value per edge")
    if not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError("conductances must be finite and positive")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be finite and non-negative")
    if not np.isfinite(q_target) or q_target <= 0.0:
        raise ValueError("q_target must be finite and positive")
    if not (0.0 < g_min < g_max):
        raise ValueError("require 0 < g_min < g_max")

    energy = edge_difference_energy(states, edges)
    signal = energy / q_target - 1.0
    if mode == "local":
        applied = signal
    elif mode == "shuffled":
        if rng is None:
            raise ValueError("shuffled mode requires an explicit RNG")
        applied = rng.permutation(signal)
    elif mode == "uniform":
        applied = np.full_like(signal, float(np.mean(signal)))
    else:
        raise ValueError(f"unknown adaptation mode: {mode}")

    proposal = g * np.exp(eta * applied)
    if not np.all(np.isfinite(proposal)):
        raise FloatingPointError("conductance update became non-finite")
    return normalize_mean_conductance(proposal, float(np.mean(g)), g_min, g_max)
