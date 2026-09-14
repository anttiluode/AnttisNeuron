import numpy as np

from anttis_neuron.branch import spectral_radius
from anttis_neuron.cable import (
    cable_operator,
    default_ports_and_sensors,
    default_tree,
    simulate_cable,
    visible_physical_modes,
    weighted_laplacian,
)


def test_default_tree_is_connected_tree():
    n, edges = default_tree()
    assert n == 11
    assert edges.shape == (10, 2)
    seen = {0}
    for _ in range(n):
        for i, j in edges:
            if int(i) in seen or int(j) in seen:
                seen.add(int(i))
                seen.add(int(j))
    assert seen == set(range(n))


def test_weighted_laplacian_is_symmetric_zero_row_sum():
    n, edges = default_tree()
    g = np.linspace(0.7, 1.3, len(edges))
    lap = weighted_laplacian(n, edges, g)
    assert np.allclose(lap, lap.T, atol=1e-12)
    assert np.allclose(lap.sum(axis=1), 0.0, atol=1e-12)


def test_default_operator_is_stable_and_observation_shape_is_bounded():
    n, edges = default_tree()
    a = cable_operator(n, edges, np.ones(len(edges)), dt=0.08, leak=0.08, coupling=0.65)
    assert spectral_radius(a) < 1.0
    b, s = default_ports_and_sensors(n)
    x = np.zeros((20, 4))
    x[0, 0] = 1.0
    z = simulate_cable(x, a, b, s)
    assert z.shape == (20, 4)
    assert np.all(np.isfinite(z))


def test_visible_modes_are_normalized_and_nontrivial():
    n, edges = default_tree()
    a = cable_operator(n, edges, np.ones(len(edges)), dt=0.08, leak=0.08, coupling=0.65)
    _, s = default_ports_and_sensors(n)
    modes = visible_physical_modes(a, s)
    assert modes.ndim == 2 and modes.shape[1] == 4
    assert modes.shape[0] >= 2
    assert np.allclose(np.linalg.norm(modes, axis=1), 1.0, atol=1e-12)
