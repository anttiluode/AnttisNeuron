"""Gate 5B: spectral decomposition audit of the frozen Gate 5 mechanism."""
from __future__ import annotations

import numpy as np

from experiments.gate5_adaptive_cable import (
    _initial_q_target,
    _make_input_tapes,
    _operator,
    _train_conditions,
)


def _uniform_index(vectors: np.ndarray) -> int:
    n = vectors.shape[0]
    uniform = np.ones(n, dtype=float) / np.sqrt(float(n))
    return int(np.argmax(np.abs(vectors.T @ uniform)))


def _full_eigendecomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("operator must be square")
    if not np.allclose(a, a.T, atol=1e-12):
        raise ValueError("Gate 5B requires a symmetric operator")
    values, vectors = np.linalg.eigh(a)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def sorted_nonuniform_eigendecomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return slowest-first eigenpairs after excluding the uniform global mode."""
    values, vectors = _full_eigendecomposition(a)
    uniform_idx = _uniform_index(vectors)
    keep = [i for i in range(len(values)) if i != uniform_idx]
    return values[keep], vectors[:, keep]


def principal_angles(q_left: np.ndarray, q_right: np.ndarray, *, rank: int = 3) -> np.ndarray:
    """Principal angles between the leading columns of two subspace bases."""
    left = np.asarray(q_left, dtype=float)
    right = np.asarray(q_right, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("subspace bases must be 2-D with matching ambient dimension")
    if rank < 1 or left.shape[1] < rank or right.shape[1] < rank:
        raise ValueError("rank must fit both subspace bases")
    q1, _ = np.linalg.qr(left[:, :rank])
    q2, _ = np.linalg.qr(right[:, :rank])
    singular = np.linalg.svd(q1.T @ q2, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return np.arccos(singular)


def hybrid_operators(a_frozen: np.ndarray, a_adapted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return spectrum-only and basis-only symmetric diagnostic operators."""
    frozen_values, frozen_vectors = _full_eigendecomposition(a_frozen)
    adapted_values, adapted_vectors = _full_eigendecomposition(a_adapted)
    if frozen_values.shape != adapted_values.shape:
        raise ValueError("operators must have the same dimension")

    a_lambda = frozen_vectors @ np.diag(adapted_values) @ frozen_vectors.T
    a_q = adapted_vectors @ np.diag(frozen_values) @ adapted_vectors.T
    a_lambda = 0.5 * (a_lambda + a_lambda.T)
    a_q = 0.5 * (a_q + a_q.T)
    return a_lambda, a_q


def build_gate5_operators(seed: int = 17) -> dict[str, np.ndarray]:
    """Reconstruct Gate 5 frozen/adapted operators without changing Gate 5."""
    adaptation_tapes, _ = _make_input_tapes(seed)
    q_target = _initial_q_target(adaptation_tapes[0])
    trained, _ = _train_conditions(adaptation_tapes, q_target=q_target, seed=seed)
    frozen_a, _, _, _ = _operator(trained["frozen"][0])
    adapted_a, _, _, _ = _operator(trained["local"][0])
    return {"frozen": frozen_a, "adapted": adapted_a}
