"""Gate 5B: spectral decomposition audit of the frozen Gate 5 mechanism."""
from __future__ import annotations

import numpy as np

from anttis_neuron.branch import spectral_radius
from anttis_neuron.cable import simulate_cable, visible_physical_modes
from anttis_neuron.oja import absolute_cosine, oja_fit
from experiments.common import main
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


def _gate5_training(seed: int) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[np.ndarray], list[np.ndarray]]:
    adaptation_tapes, heldout_tapes = _make_input_tapes(seed)
    q_target = _initial_q_target(adaptation_tapes[0])
    trained, _ = _train_conditions(adaptation_tapes, q_target=q_target, seed=seed)
    return trained, adaptation_tapes, heldout_tapes


def build_gate5_operators(seed: int = 17) -> dict[str, np.ndarray]:
    """Reconstruct Gate 5 frozen/adapted operators without changing Gate 5."""
    trained, _, _ = _gate5_training(seed)
    frozen_a, _, _, _ = _operator(trained["frozen"][0])
    adapted_a, _, _, _ = _operator(trained["local"][0])
    return {"frozen": frozen_a, "adapted": adapted_a}


def _evaluate_operator(
    a: np.ndarray,
    tapes: list[np.ndarray],
    b: np.ndarray,
    s: np.ndarray,
    *,
    oja_seed_base: int,
) -> tuple[dict, list[np.ndarray]]:
    physical_modes = visible_physical_modes(a, s)[:3]
    if len(physical_modes) < 3:
        raise FloatingPointError("Gate 5B requires three visible non-uniform modes")

    alignments: list[float] = []
    learned_vectors: list[np.ndarray] = []
    for index, tape in enumerate(tapes):
        observations = simulate_cable(tape, a, b, s, burn=250)
        rms = float(np.sqrt(np.mean(np.sum(observations * observations, axis=1))))
        if rms <= 0.0 or not np.isfinite(rms):
            raise FloatingPointError("invalid observation RMS")
        learned = oja_fit(observations / rms, lr=0.001, epochs=3, seed=oja_seed_base + index)
        learned_vectors.append(learned)
        alignments.append(max(absolute_cosine(learned, mode) for mode in physical_modes))

    return (
        {
            "alignments": [float(v) for v in alignments],
            "mean_alignment": float(np.mean(alignments)),
            "median_alignment": float(np.median(alignments)),
            "min_alignment": float(np.min(alignments)),
            "spectral_radius": spectral_radius(a),
        },
        learned_vectors,
    )


def _visible_slow_modes(a: np.ndarray, s: np.ndarray) -> np.ndarray:
    modes = visible_physical_modes(a, s)[:3]
    if len(modes) < 3:
        raise FloatingPointError("expected three visible slow modes")
    return modes


def _mean_cross_alignment(vectors: list[np.ndarray], modes: np.ndarray) -> float:
    values = [max(absolute_cosine(vector, mode) for mode in modes) for vector in vectors]
    return float(np.mean(values))


def run(seed: int = 17) -> dict:
    trained, _, heldout_tapes = _gate5_training(seed)
    a_frozen, b, s, _ = _operator(trained["frozen"][0])
    a_adapted, _, _, _ = _operator(trained["local"][0])
    a_lambda, a_q = hybrid_operators(a_frozen, a_adapted)

    frozen_values, frozen_vectors = sorted_nonuniform_eigendecomposition(a_frozen)
    adapted_values, adapted_vectors = sorted_nonuniform_eigendecomposition(a_adapted)
    angles = principal_angles(frozen_vectors, adapted_vectors, rank=3)

    state_overlap = np.abs(frozen_vectors[:, :3].T @ adapted_vectors[:, :3])
    slow_best = np.max(state_overlap, axis=1)

    frozen_visible = _visible_slow_modes(a_frozen, s)
    adapted_visible = _visible_slow_modes(a_adapted, s)
    visible_overlap = np.abs(frozen_visible @ adapted_visible.T)

    diagnostic_operators = {
        "frozen": a_frozen,
        "eigenvalue_only": a_lambda,
        "eigenvector_only": a_q,
        "adapted": a_adapted,
    }
    diagnostic_conditions: dict[str, dict] = {}
    learned: dict[str, list[np.ndarray]] = {}
    for name, operator in diagnostic_operators.items():
        summary, vectors = _evaluate_operator(
            operator,
            heldout_tapes,
            b,
            s,
            oja_seed_base=seed + 3000,
        )
        diagnostic_conditions[name] = summary
        learned[name] = vectors

    cross_basis = {
        "frozen_oja_vs_frozen_modes": _mean_cross_alignment(learned["frozen"], frozen_visible),
        "frozen_oja_vs_adapted_modes": _mean_cross_alignment(learned["frozen"], adapted_visible),
        "adapted_oja_vs_frozen_modes": _mean_cross_alignment(learned["adapted"], frozen_visible),
        "adapted_oja_vs_adapted_modes": _mean_cross_alignment(learned["adapted"], adapted_visible),
    }

    return {
        "gate": "5B",
        "seed": seed,
        "frozen_slow_nonuniform_eigenvalues": [float(v) for v in frozen_values[:3]],
        "adapted_slow_nonuniform_eigenvalues": [float(v) for v in adapted_values[:3]],
        "slow_subspace_principal_angles_degrees": [float(v) for v in np.degrees(angles)],
        "slow_mode_best_match_overlaps": [float(v) for v in slow_best],
        "state_slow_mode_overlap_matrix": [[float(v) for v in row] for row in state_overlap],
        "visible_mode_overlap_matrix": [[float(v) for v in row] for row in visible_overlap],
        "diagnostic_conditions": diagnostic_conditions,
        "cross_basis_mean_alignments": cross_basis,
        "interpretation": (
            "diagnostic decomposition only; eigenvalue-only and eigenvector-only hybrids are "
            "symmetric counterfactual operators, not claimed realizable positive-conductance trees"
        ),
    }


if __name__ == "__main__":
    main(run)
