import numpy as np

from anttis_neuron.branch import spectral_radius
from anttis_neuron.cable import cable_operator, visible_physical_modes
from anttis_neuron.worlds import generate_world, port_sensor_matrices


def _connected(n_nodes, edges):
    seen = {0}
    for _ in range(n_nodes):
        for i, j in edges:
            if int(i) in seen or int(j) in seen:
                seen.add(int(i))
                seen.add(int(j))
    return seen == set(range(n_nodes))


def test_world_generation_is_deterministic_and_indexed():
    a = generate_world(3, master_seed=1701)
    b = generate_world(3, master_seed=1701)
    c = generate_world(4, master_seed=1701)
    assert a == b
    assert a.index == 3 and c.index == 4
    assert (a.edges != c.edges).any() or a.port_nodes != c.port_nodes or a.sensor_nodes != c.sensor_nodes


def test_24_worlds_are_valid_stable_branched_observers():
    for index in range(24):
        world = generate_world(index, master_seed=1701)
        assert world.n_nodes == 11
        assert world.edges.shape == (10, 2)
        assert _connected(world.n_nodes, world.edges)
        assert len(set(world.port_nodes)) == 4
        assert len(set(world.sensor_nodes)) == 4
        assert all(1 <= node < 11 for node in world.port_nodes)
        assert all(1 <= node < 11 for node in world.sensor_nodes)
        b, s = port_sensor_matrices(world)
        assert b.shape == (11, 4)
        assert s.shape == (4, 11)
        a = cable_operator(11, world.edges, np.ones(10), dt=0.08, leak=0.08, coupling=0.65)
        assert spectral_radius(a) < 1.0
        assert visible_physical_modes(a, s).shape[0] >= 3


def test_world_covariance_schedule_has_four_adaptation_and_five_distinct_heldout_angles():
    world = generate_world(7, master_seed=1701)
    assert len(world.adaptation_angles) == 4
    assert len(world.heldout_angles) == 5
    assert len(world.adaptation_spectra) == 4
    assert len(world.heldout_spectrum) == 4
    for heldout in world.heldout_angles:
        assert all(abs(heldout - train) > 1e-6 for train in world.adaptation_angles)
