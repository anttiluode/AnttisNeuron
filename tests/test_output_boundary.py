import math

import numpy as np

from anttis_neuron.output_boundary import (
    calibrate_threshold,
    driving_point_conductance,
    homeostatic_gain,
    rate_boundary,
    steady_state_soma_voltage,
    transfer_metrics,
)


def test_two_node_load_matches_schur_complement():
    edges = np.array([[0, 1]], dtype=int)
    conductances = np.array([2.0])
    got = driving_point_conductance(
        2,
        edges,
        conductances,
        leak=0.08,
        coupling=0.65,
    )
    diag = 0.08 + 0.65 * 2.0
    off = -0.65 * 2.0
    expected = diag - off * off / diag
    assert math.isclose(got, expected, rel_tol=0.0, abs_tol=1e-12)
    assert got > 0.0


def test_steady_state_soma_voltage_is_current_over_load():
    current = np.array([0.5, 1.0, 1.5])
    voltage = steady_state_soma_voltage(current, load=0.25)
    assert np.allclose(voltage, np.array([2.0, 4.0, 6.0]), rtol=0.0, atol=1e-15)


def test_schur_load_voltage_matches_direct_graph_solve():
    edges = np.array([[0, 1]], dtype=int)
    conductances = np.array([2.0])
    leak = 0.08
    coupling = 0.65
    load = driving_point_conductance(
        2,
        edges,
        conductances,
        leak=leak,
        coupling=coupling,
    )
    current = 1.7
    from_load = float(steady_state_soma_voltage(np.array([current]), load=load)[0])

    lap = np.array([[2.0, -2.0], [-2.0, 2.0]])
    conductance_matrix = leak * np.eye(2) + coupling * lap
    direct = np.linalg.solve(conductance_matrix, np.array([current, 0.0]))
    assert math.isclose(from_load, float(direct[0]), rel_tol=0.0, abs_tol=1e-12)


def test_steady_state_soma_voltage_rejects_nonpositive_load():
    for load in (0.0, -1.0):
        try:
            steady_state_soma_voltage(np.array([1.0]), load=load)
        except ValueError:
            pass
        else:
            raise AssertionError("steady-state soma voltage must reject nonpositive load")


def test_rate_boundary_is_finite_and_bounded():
    voltage = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
    rate = rate_boundary(voltage, gain=1.2, theta=0.1, beta=3.0)
    assert np.all(np.isfinite(rate))
    assert np.all(rate >= 0.0)
    assert np.all(rate <= 1.0)
    assert np.all(np.diff(rate) >= 0.0)


def test_threshold_calibration_hits_target_rate():
    voltage = np.linspace(-1.0, 1.0, 2001)
    theta = calibrate_threshold(
        voltage,
        gain=1.0,
        beta=2.0,
        target_rate=0.25,
    )
    rate = rate_boundary(voltage, gain=1.0, theta=theta, beta=2.0)
    assert abs(float(np.mean(rate)) - 0.25) < 1e-10


def test_homeostatic_gain_moves_toward_target_and_clips():
    assert homeostatic_gain(1.0, 0.10, target_rate=0.25) > 1.0
    assert homeostatic_gain(1.0, 0.40, target_rate=0.25) < 1.0
    assert homeostatic_gain(5.0, 0.0, target_rate=1.0) <= 5.0
    assert homeostatic_gain(0.2, 1.0, target_rate=0.0) >= 0.2


def test_transfer_metrics_self_rmse_rheobase_and_gain():
    scales = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
    rates = np.array([0.04, 0.08, 0.16, 0.26, 0.38])
    result = transfer_metrics(scales, rates, rates, rheobase_rate=0.10)

    assert result["curve_rmse"] == 0.0
    assert math.isclose(result["rheobase"], 0.8125, rel_tol=0.0, abs_tol=1e-12)
    expected_slope = float(np.polyfit(scales[1:4], rates[1:4], 1)[0])
    assert math.isclose(result["fi_gain"], expected_slope, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result["unit_rate"], 0.16, rel_tol=0.0, abs_tol=1e-12)


def test_transfer_metrics_missing_rheobase_is_none():
    scales = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
    rates = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    reference = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    result = transfer_metrics(scales, rates, reference, rheobase_rate=0.10)
    assert result["rheobase"] is None
