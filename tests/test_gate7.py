import inspect
import math

import numpy as np

from anttis_neuron.output_boundary import homeostatic_gain
from anttis_neuron.worlds import generate_world
from experiments.gate6_many_worlds import trained_world_conductances
from experiments.gate7_load_compensation import (
    SCALES,
    _make_gate7_tapes,
    _reference_parameters,
    run_world,
)


def test_gate7_tapes_are_deterministic_standard_gaussian_probes():
    world = generate_world(0, master_seed=1718)
    cal_a, eval_a = _make_gate7_tapes(world)
    cal_b, eval_b = _make_gate7_tapes(world)

    assert cal_a.shape == (3000, 4)
    assert eval_a.shape == (5000, 4)
    assert np.array_equal(cal_a, cal_b)
    assert np.array_equal(eval_a, eval_b)
    assert np.all(np.isfinite(cal_a))
    assert np.all(np.isfinite(eval_a))
    assert np.max(np.abs(np.mean(eval_a, axis=0))) < 0.08
    assert np.max(np.abs(np.cov(eval_a, rowvar=False) - np.eye(4))) < 0.08


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


def test_gate7_reference_curve_is_nontrivial_and_predeclared():
    world = generate_world(0, master_seed=1718)
    trained = trained_world_conductances(world, tape_seed=6017)
    reference = _reference_parameters(world, trained["local"])

    assert np.array_equal(SCALES, np.array([0.50, 0.75, 1.00, 1.25, 1.50]))
    assert reference["target_rate"] == 0.25
    assert reference["reference_load"] > 0.0
    assert reference["beta"] > 0.0
    assert math.isfinite(reference["theta"])
    curve = np.asarray(reference["reference_curve"], dtype=float)
    assert curve.shape == (5,)
    assert np.max(curve) - np.min(curve) >= 0.10
    assert abs(reference["reference_unit_calibration_rate"] - 0.25) < 1e-10


def test_gate7_world_conditions_share_probe_and_obey_boundaries():
    reference_world = generate_world(0, master_seed=1718)
    reference_g = trained_world_conductances(reference_world, tape_seed=6017)["local"]
    reference = _reference_parameters(reference_world, reference_g)

    world = generate_world(1, master_seed=1718)
    result = run_world(world, reference=reference)

    assert result["fixed_gain_start"] == 1.0
    assert result["homeostatic_gain_start"] == 1.0
    assert math.isclose(
        result["oracle_gain"],
        result["load"] / result["reference_load"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert len(result["calibration_tape_digest"]) == 64
    assert len(result["evaluation_tape_digest"]) == 64
    assert result["scales"] == [0.5, 0.75, 1.0, 1.25, 1.5]

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

    assert -1.0 <= result["homeostatic_delta"] <= 1.0
    assert -1.0 <= result["oracle_delta"] <= 1.0
