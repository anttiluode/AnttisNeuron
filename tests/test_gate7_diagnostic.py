"""Temporary diagnostic for Gate 7 reference-curve assay calibration."""
import numpy as np

from anttis_neuron.output_boundary import calibrate_threshold, rate_boundary
from anttis_neuron.worlds import generate_world
from experiments.gate6_many_worlds import trained_world_conductances
from experiments.gate7_load_compensation import (
    SCALES,
    _make_gate7_tapes,
    _soma_voltage,
)


def test_gate7_reference_sensitivity_diagnostic():
    world = generate_world(0, master_seed=1718)
    conductances = trained_world_conductances(world, tape_seed=6017)["local"]
    calibration, evaluation = _make_gate7_tapes(world)
    cal_v = _soma_voltage(world, conductances, calibration)
    std = float(np.std(cal_v))

    report = {}
    for coefficient in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        beta = coefficient / std
        theta = calibrate_threshold(
            cal_v,
            gain=1.0,
            beta=beta,
            target_rate=0.25,
        )
        curve = []
        for scale in SCALES:
            voltage = _soma_voltage(world, conductances, evaluation, scale=float(scale))
            curve.append(float(np.mean(rate_boundary(voltage, gain=1.0, theta=theta, beta=beta))))
        report[coefficient] = {
            "beta": beta,
            "theta": theta,
            "curve": curve,
            "dynamic_range": max(curve) - min(curve),
        }

    raise AssertionError(report)
