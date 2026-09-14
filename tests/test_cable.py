import inspect

import numpy as np

from anttis_neuron.branch import spectral_radius
from anttis_neuron.cable import (
    adapt_conductances,
    cable_operator,
    default_ports_and_sensors,
    default_tree,
    edge_difference_energy,
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


def test_simulation_can_return_full_states_for_local_plasticity():
    n, edges = default_tree()
    a = cable_operator(n, edges, np.ones(len(edges)), dt=0.08, leak=0.08, coupling=0.65)
    b, s = default_ports_and_sensors(n)
    x = np.zeros((13, 4))
    z, states = simulate_cable(x, a, b, s, return_states=True)
    assert z.shape == (13, 4)
    assert states.shape == (13, n)


def test_visible_modes_are_normalized_and_nontrivial():
    n, edges = default_tree()
    a = cable_operator(n, edges, np.ones(len(edges)), dt=0.08, leak=0.08, coupling=0.65)
    _, s = default_ports_and_sensors(n)
    modes = visible_physical_modes(a, s)
    assert modes.ndim == 2 and modes.shape[1] == 4
    assert modes.shape[0] >= 2
    assert np.allclose(np.linalg.norm(modes, axis=1), 1.0, atol=1e-12)


def test_edge_difference_energy_matches_manual_two_edge_example():
    states = np.array([[0.0, 1.0, 3.0], [2.0, 1.0, 1.0]])
    edges = np.array([[0, 1], [1, 2]])
    energy = edge_difference_energy(states, edges)
    assert np.allclose(energy, np.array([1.0, 2.0]))


def test_local_adaptor_signature_has_no_alignment_or_mode_argument():
    names = set(inspect.signature(adapt_conductances).parameters)
    forbidden = {"alignment", "score", "eigenvectors", "physical_modes", "oja", "labels"}
    assert names.isdisjoint(forbidden)


def test_mean_normalization_and_bounds_are_preserved():
    n, edges = default_tree()
    rng = np.random.default_rng(4)
    states = rng.normal(size=(200, n))
    g0 = np.ones(len(edges))
    g1 = adapt_conductances(
        states,
        edges,
        g0,
        eta=0.08,
        q_target=1.0,
        g_min=0.45,
        g_max=1.8,
    )
    assert np.all(np.isfinite(g1))
    assert np.all((g1 >= 0.45) & (g1 <= 1.8))
    assert np.isclose(g1.mean(), g0.mean(), atol=1e-10)
    assert np.std(g1) > 0.0


def test_uniform_control_preserves_relative_conductances():
    n, edges = default_tree()
    rng = np.random.default_rng(5)
    states = rng.normal(size=(100, n))
    g0 = np.linspace(0.8, 1.2, len(edges))
    g1 = adapt_conductances(
        states,
        edges,
        g0,
        eta=0.1,
        q_target=1.0,
        g_min=0.4,
        g_max=2.0,
        mode="uniform",
    )
    assert np.allclose(g1 / g1.mean(), g0 / g0.mean(), atol=1e-12)


def test_shuffled_control_is_deterministic_for_fixed_rng():
    n, edges = default_tree()
    rng_states = np.random.default_rng(6)
    states = rng_states.normal(size=(150, n))
    g0 = np.ones(len(edges))
    a = adapt_conductances(
        states,
        edges,
        g0,
        eta=0.05,
        q_target=1.0,
        g_min=0.4,
        g_max=2.0,
        mode="shuffled",
        rng=np.random.default_rng(9),
    )
    b = adapt_conductances(
        states,
        edges,
        g0,
        eta=0.05,
        q_target=1.0,
        g_min=0.4,
        g_max=2.0,
        mode="shuffled",
        rng=np.random.default_rng(9),
    )
    assert np.allclose(a, b)
