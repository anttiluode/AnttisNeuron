import math

import numpy as np

from anttis_neuron.branch import spectral_radius
from experiments.gate5b_spectral_audit import (
    build_gate5_operators,
    hybrid_operators,
    principal_angles,
    run,
    sorted_nonuniform_eigendecomposition,
)


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def test_principal_angles_identity_are_zero():
    q = np.eye(5)[:, :3]
    assert np.allclose(principal_angles(q, q, rank=3), 0.0, atol=1e-12)


def test_nonuniform_eigendecomposition_excludes_uniform_mode():
    operators = build_gate5_operators(seed=17)
    values, vectors = sorted_nonuniform_eigendecomposition(operators["frozen"])
    assert values.shape == (10,)
    assert vectors.shape == (11, 10)
    assert np.all(np.diff(values) <= 1e-12)
    uniform = np.ones(11) / np.sqrt(11.0)
    assert np.max(np.abs(vectors.T @ uniform)) < 1e-8


def test_hybrid_operator_reconstruction_parts():
    operators = build_gate5_operators(seed=17)
    a_frozen = operators["frozen"]
    a_adapted = operators["adapted"]
    a_lambda, a_q = hybrid_operators(a_frozen, a_adapted)

    frozen_eigenvalues = np.linalg.eigvalsh(a_frozen)
    adapted_eigenvalues = np.linalg.eigvalsh(a_adapted)
    assert np.allclose(np.sort(np.linalg.eigvalsh(a_lambda)), np.sort(adapted_eigenvalues), atol=1e-12)
    assert np.allclose(np.sort(np.linalg.eigvalsh(a_q)), np.sort(frozen_eigenvalues), atol=1e-12)
    assert np.allclose(a_lambda, a_lambda.T, atol=1e-12)
    assert np.allclose(a_q, a_q.T, atol=1e-12)


def test_all_gate5b_diagnostic_operators_are_stable():
    operators = build_gate5_operators(seed=17)
    a_lambda, a_q = hybrid_operators(operators["frozen"], operators["adapted"])
    for a in (operators["frozen"], a_lambda, a_q, operators["adapted"]):
        assert spectral_radius(a) < 1.0


def test_gate5b_receipt_is_deterministic_finite_and_bounded():
    a = run(seed=17)
    b = run(seed=17)
    assert a == b
    assert _finite(a)
    assert a["gate"] == "5B"
    assert a["seed"] == 17
    assert len(a["slow_subspace_principal_angles_degrees"]) == 3
    assert len(a["slow_mode_best_match_overlaps"]) == 3
    assert len(a["visible_mode_overlap_matrix"]) == 3
    assert all(len(row) == 3 for row in a["visible_mode_overlap_matrix"])
    for value in a["slow_mode_best_match_overlaps"]:
        assert 0.0 <= value <= 1.0
    for row in a["visible_mode_overlap_matrix"]:
        for value in row:
            assert 0.0 <= value <= 1.0
    for name in ("frozen", "eigenvalue_only", "eigenvector_only", "adapted"):
        result = a["diagnostic_conditions"][name]
        assert len(result["alignments"]) == 5
        assert 0.0 <= result["mean_alignment"] <= 1.0
        assert 0.0 <= result["median_alignment"] <= 1.0
        assert 0.0 <= result["min_alignment"] <= 1.0
        assert result["spectral_radius"] < 1.0
    for value in a["cross_basis_mean_alignments"].values():
        assert 0.0 <= value <= 1.0
