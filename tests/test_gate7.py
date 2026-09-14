import inspect
import math

import numpy as np

from anttis_neuron.output_boundary import homeostatic_gain, rate_boundary
from anttis_neuron.worlds import generate_world
from experiments.gate6_many_worlds import trained_world_conductances
from experiments.gate7_load_compensation import (
    CURRENTS,
    _reference_parameters,
    run_world,
)


def _reference_fixture():
    world = generate_world(0, master_seed=1718)
    conductances = trained_world_conductances(world, tape_seed=6017)["local"]
    return world, conductances, _reference_parameters(world, conductances)


def test_homeostatic_rule_has_no_structural_or_load_inputs():
    parameters = set(inspect.signature(homeostatic_gain).parameters)
    assert parameters == {
        "gain",
        "mean_rate",
        "target_rate",
        "eta",
        "gain_min",
        "gain_max",
    }
    forbidden = {"load", "world", "topology", "conductances", "eigenmodes", "alignment"}
    assert parameters.isdisjoint(forbidden)


def test_gate7_reference_is_analytic_nontrivial_somatic_current_curve():
    _, _, reference = _reference_fixture()

    assert np.array_equal(CURRENTS, np.array([0.50, 0.75, 1.00, 1.25, 1.50]))
    assert reference["target_rate"] == 0.25
    load = float(reference["reference_load"])
    assert math.isfinite(load) and load > 0.0

    v_star = 1.0 / load
    expected_beta = 4.0 / v_star
    expected_theta = v_star - math.log(0.25 / 0.75) / expected_beta
    assert math.isclose(reference["beta"], expected_beta, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(reference["theta"], expected_theta, rel_tol=0.0, abs_tol=1e-14)

    unit_rate = float(
        rate_boundary(
            np.array([v_star]),
            gain=1.0,
            theta=reference["theta"],
            beta=reference["beta"],
        )[0]
    )
    assert math.isclose(unit_rate, 0.25, rel_tol=0.0, abs_tol=1e-14)
    assert math.isclose(reference["reference_unit_rate"], 0.25, rel_tol=0.0, abs_tol=1e-14)

    curve = np.asarray(reference["reference_curve"], dtype=float)
    assert curve.shape == (5,)
    assert np.all(np.diff(curve) > 0.0)
    assert float(np.max(curve) - np.min(curve)) >= 0.10
    assert math.isclose(
        reference["reference_curve_dynamic_range"],
        float(np.max(curve) - np.min(curve)),
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_gate7_oracle_exactly_cancels_somatic_load():
    _, _, reference = _reference_fixture()
    world = generate_world(1, master_seed=1718)
    result = run_world(world, reference=reference)

    assert result["fixed_gain_start"] == 1.0
    assert result["homeostatic_gain_start"] == 1.0
    assert math.isfinite(result["load"]) and result["load"] > 0.0
    assert math.isclose(
        result["oracle_gain"],
        result["load"] / result["reference_load"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert result["currents"] == [0.5, 0.75, 1.0, 1.25, 1.5]

    reference_curve = np.asarray(reference["reference_curve"], dtype=float)
    oracle_curve = np.asarray(result["oracle"]["rates"], dtype=float)
    assert np.allclose(oracle_curve, reference_curve, rtol=0.0, atol=2e-14)
    assert result["oracle"]["metrics"]["curve_rmse"] < 2e-14
    assert result["oracle_numerical_error"] < 2e-14

    for condition in ("fixed", "homeostatic", "oracle"):
        rates = np.asarray(result[condition]["rates"], dtype=float)
        assert rates.shape == (5,)
        assert np.all(np.isfinite(rates))
        assert np.all((rates >= 0.0) & (rates <= 1.0))
        metrics = result[condition]["metrics"]
        assert metrics["curve_rmse"] >= 0.0
        assert math.isfinite(metrics["fi_gain"])
        assert 0.0 <= metrics["unit_rate"] <= 1.0
        assert isinstance(result[condition]["collapsed"], bool)

    # Scientific sign is deliberately unconstrained: negative homeostatic delta
    # remains a valid frozen result.
    assert -1.0 <= result["homeostatic_delta"] <= 1.0


def test_gate7_reference_world_oracle_and_fixed_are_identical():
    world, _, reference = _reference_fixture()
    result = run_world(world, reference=reference)
    fixed = np.asarray(result["fixed"]["rates"], dtype=float)
    oracle = np.asarray(result["oracle"]["rates"], dtype=float)
    assert math.isclose(result["oracle_gain"], 1.0, rel_tol=0.0, abs_tol=1e-15)
    assert np.allclose(fixed, oracle, rtol=0.0, atol=2e-14)
