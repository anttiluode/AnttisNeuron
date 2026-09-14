import numpy as np

from anttis_neuron.branch import spectral_radius
from experiments.gate5b_spectral_audit import (
    build_gate5_operators,
    hybrid_operators,
    principal_angles,
    sorted_nonuniform_eigendecomposition,
)


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
