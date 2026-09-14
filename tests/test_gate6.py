import math

import numpy as np

from anttis_neuron.worlds import generate_world
from experiments.gate6_many_worlds import _subspace_alignment, run, run_world


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


def test_gate6_subspace_alignment_is_basis_rotation_invariant():
    basis = np.eye(4)[:, :3]
    theta = 0.731
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = basis @ rotation
    vector = np.array([0.2, -0.5, 0.3, 0.7])

    score = _subspace_alignment(vector, basis)
    rotated_score = _subspace_alignment(vector, rotated)

    assert math.isclose(score, rotated_score, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(score, np.linalg.norm(vector[:3]) / np.linalg.norm(vector), abs_tol=1e-12)


def test_gate6_two_world_smoke_is_deterministic_and_finite():
    a = run(seed=17, n_worlds=2)
    b = run(seed=17, n_worlds=2)
    assert a == b
    assert _finite(a)
    assert a["gate"] == 6
    assert a["seed"] == 17
    assert a["n_worlds"] == 2
    assert len(a["worlds"]) == 2


def test_gate6_world_controls_are_matched_stable_and_bounded():
    result = run(seed=17, n_worlds=2)
    for world in result["worlds"]:
        assert world["common_tape_digest"] == world["control_tape_digest"]
        assert world["control_update_multiset_max_error"] < 1e-12
        assert math.isclose(
            world["uniform_heldout_mean_alignment"],
            world["frozen_heldout_mean_alignment"],
            abs_tol=1e-12,
        )
        for key in (
            "frozen_heldout_mean_alignment",
            "adaptive_heldout_mean_alignment",
            "shuffled_heldout_mean_alignment",
            "uniform_heldout_mean_alignment",
            "frozen_heldout_min_alignment",
            "adaptive_heldout_min_alignment",
            "shuffled_heldout_min_alignment",
        ):
            assert 0.0 <= world[key] <= 1.0
        assert -1.0 <= world["adaptive_minus_frozen"] <= 1.0
        assert -1.0 <= world["adaptive_minus_shuffled"] <= 1.0
        for key in (
            "frozen_spectral_radius",
            "adaptive_spectral_radius",
            "shuffled_spectral_radius",
            "uniform_spectral_radius",
        ):
            assert world[key] < 1.0
        assert len(world["topology_edges"]) == 10
        assert len(world["port_nodes"]) == 4
        assert len(world["sensor_nodes"]) == 4


def test_gate6_known_near_degenerate_world_keeps_uniform_control_exactly_frozen():
    world = generate_world(4, master_seed=1718)
    result = run_world(world, tape_seed=6017)
    assert result["uniform_heldout_mean_alignment"] == result["frozen_heldout_mean_alignment"]
    assert result["uniform_spectral_radius"] == result["frozen_spectral_radius"]


def test_gate6_aggregate_reports_distribution_without_positive_requirement():
    result = run(seed=17, n_worlds=2)
    aggregate = result["aggregate"]
    for key in (
        "adaptive_minus_frozen_mean",
        "adaptive_minus_frozen_median",
        "adaptive_minus_frozen_q25",
        "adaptive_minus_frozen_min",
        "adaptive_minus_shuffled_mean",
        "adaptive_minus_shuffled_median",
        "adaptive_minus_shuffled_q25",
        "adaptive_minus_shuffled_min",
    ):
        assert -1.0 <= aggregate[key] <= 1.0
    for key in (
        "adaptive_beats_frozen_fraction",
        "adaptive_beats_shuffled_fraction",
    ):
        assert 0.0 <= aggregate[key] <= 1.0
    assert 0 <= aggregate["adaptive_loses_to_frozen_count"] <= 2
    assert 0 <= aggregate["adaptive_loses_to_shuffled_count"] <= 2


def test_gate6_rejects_world_count_outside_predeclared_suite():
    for n_worlds in (0, 25):
        try:
            run(seed=17, n_worlds=n_worlds)
        except ValueError:
            pass
        else:
            raise AssertionError("Gate 6 must enforce 1 <= n_worlds <= 24")
