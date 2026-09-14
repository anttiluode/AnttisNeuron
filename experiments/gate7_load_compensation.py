"""Gate 7: local output-boundary compensation for somatic dendritic load."""
from __future__ import annotations

import math

import numpy as np

from anttis_neuron.output_boundary import (
    driving_point_conductance,
    homeostatic_gain,
    rate_boundary,
    steady_state_soma_voltage,
    transfer_metrics,
)
from anttis_neuron.worlds import CableWorld
from experiments.gate5_adaptive_cable import _COUPLING, _LEAK
from experiments.gate6_many_worlds import trained_world_conductances


CURRENTS = np.asarray((0.50, 0.75, 1.00, 1.25, 1.50), dtype=float)
_TARGET_RATE = 0.25
_REFERENCE_SLOPE_COEFFICIENT = 4.0
_HOMEOSTATIC_PHASES = 12
_HOMEOSTATIC_ETA = 0.5
_GAIN_MIN = 0.2
_GAIN_MAX = 5.0
_DENDRITIC_TAPE_SEED = 6017
_COLLAPSE_LOW = 0.02
_COLLAPSE_HIGH = 0.98


def _world_load(world: CableWorld, conductances: np.ndarray) -> float:
    return driving_point_conductance(
        world.n_nodes,
        world.edges,
        conductances,
        leak=_LEAK,
        coupling=_COUPLING,
        soma_node=0,
    )


def _scalar_rate(voltage: float, *, gain: float, theta: float, beta: float) -> float:
    value = rate_boundary(
        np.asarray([voltage], dtype=float),
        gain=gain,
        theta=theta,
        beta=beta,
    )
    return float(value[0])


def _rate_curve(load: float, *, gain: float, theta: float, beta: float) -> np.ndarray:
    voltage = steady_state_soma_voltage(CURRENTS, load=load)
    rates = rate_boundary(voltage, gain=gain, theta=theta, beta=beta)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0.0) or np.any(rates > 1.0):
        raise FloatingPointError("invalid Gate 7 transfer curve")
    return rates


def _adapt_homeostatic_gain(
    unit_voltage: float,
    *,
    theta: float,
    beta: float,
    target_rate: float,
) -> tuple[float, list[float]]:
    """Adapt output gain using only local output-rate error at unit current."""
    if not np.isfinite(unit_voltage) or unit_voltage <= 0.0:
        raise ValueError("unit_voltage must be finite and positive")
    gain = 1.0
    rate_history: list[float] = []
    for _ in range(_HOMEOSTATIC_PHASES):
        rate = _scalar_rate(unit_voltage, gain=gain, theta=theta, beta=beta)
        rate_history.append(rate)
        gain = homeostatic_gain(
            gain,
            rate,
            target_rate=target_rate,
            eta=_HOMEOSTATIC_ETA,
            gain_min=_GAIN_MIN,
            gain_max=_GAIN_MAX,
        )
    return float(gain), [float(value) for value in rate_history]


def _reference_parameters(
    reference_world: CableWorld,
    reference_g: np.ndarray,
) -> dict:
    """Analytically freeze the output nonlinearity from reference-world load."""
    reference_load = _world_load(reference_world, reference_g)
    unit_voltage = 1.0 / reference_load
    beta = _REFERENCE_SLOPE_COEFFICIENT / unit_voltage
    target_logit = math.log(_TARGET_RATE / (1.0 - _TARGET_RATE))
    theta = unit_voltage - target_logit / beta

    reference_curve = _rate_curve(
        reference_load,
        gain=1.0,
        theta=theta,
        beta=beta,
    )
    dynamic_range = float(np.max(reference_curve) - np.min(reference_curve))
    if dynamic_range < 0.10:
        raise FloatingPointError(
            f"reference transfer curve is too flat: dynamic range={dynamic_range}"
        )
    unit_index = int(np.argmin(np.abs(CURRENTS - 1.0)))
    unit_rate = float(reference_curve[unit_index])
    if not math.isclose(unit_rate, _TARGET_RATE, rel_tol=0.0, abs_tol=1e-13):
        raise FloatingPointError("analytic Gate 7 reference calibration missed target rate")

    return {
        "reference_world_index": int(reference_world.index),
        "reference_world_seed": int(reference_world.seed),
        "target_rate": _TARGET_RATE,
        "slope_coefficient": _REFERENCE_SLOPE_COEFFICIENT,
        "beta": float(beta),
        "theta": float(theta),
        "reference_load": float(reference_load),
        "reference_unit_voltage": float(unit_voltage),
        "reference_curve": [float(value) for value in reference_curve],
        "reference_curve_dynamic_range": dynamic_range,
        "reference_unit_rate": unit_rate,
    }


def _condition_result(
    rates: np.ndarray,
    reference_curve: np.ndarray,
    *,
    gain: float,
) -> dict:
    metrics = transfer_metrics(CURRENTS, rates, reference_curve, rheobase_rate=0.10)
    collapsed = bool(
        metrics["unit_rate"] < _COLLAPSE_LOW
        or metrics["unit_rate"] > _COLLAPSE_HIGH
    )
    return {
        "gain": float(gain),
        "rates": [float(value) for value in rates],
        "metrics": metrics,
        "collapsed": collapsed,
    }


def run_world(world: CableWorld, *, reference: dict) -> dict:
    """Evaluate fixed, local-homeostatic, and exact load-oracle boundaries."""
    trained = trained_world_conductances(world, tape_seed=_DENDRITIC_TAPE_SEED)
    conductances = trained["local"]
    load = _world_load(world, conductances)

    theta = float(reference["theta"])
    beta = float(reference["beta"])
    target_rate = float(reference["target_rate"])
    reference_load = float(reference["reference_load"])
    reference_curve = np.asarray(reference["reference_curve"], dtype=float)

    fixed_gain = 1.0
    unit_voltage = float(steady_state_soma_voltage(np.asarray([1.0]), load=load)[0])
    local_gain, local_rate_history = _adapt_homeostatic_gain(
        unit_voltage,
        theta=theta,
        beta=beta,
        target_rate=target_rate,
    )
    oracle_gain = float(load / reference_load)

    fixed_rates = _rate_curve(load, gain=fixed_gain, theta=theta, beta=beta)
    local_rates = _rate_curve(load, gain=local_gain, theta=theta, beta=beta)
    oracle_rates = _rate_curve(load, gain=oracle_gain, theta=theta, beta=beta)

    fixed = _condition_result(fixed_rates, reference_curve, gain=fixed_gain)
    local = _condition_result(local_rates, reference_curve, gain=local_gain)
    oracle = _condition_result(oracle_rates, reference_curve, gain=oracle_gain)
    oracle_error = float(np.max(np.abs(oracle_rates - reference_curve)))

    return {
        "world_index": int(world.index),
        "world_seed": int(world.seed),
        "topology_edges": [[int(i), int(j)] for i, j in world.edges],
        "load": float(load),
        "reference_load": reference_load,
        "load_ratio": float(load / reference_load),
        "unit_voltage": unit_voltage,
        "currents": [float(value) for value in CURRENTS],
        "fixed_gain_start": 1.0,
        "homeostatic_gain_start": 1.0,
        "oracle_gain": oracle_gain,
        "homeostatic_rate_history": local_rate_history,
        "fixed": fixed,
        "homeostatic": local,
        "oracle": oracle,
        "homeostatic_delta": float(
            fixed["metrics"]["curve_rmse"] - local["metrics"]["curve_rmse"]
        ),
        "oracle_numerical_error": oracle_error,
        "homeostatic_gain_minus_oracle": float(local_gain - oracle_gain),
    }
