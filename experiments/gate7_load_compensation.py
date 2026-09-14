"""Gate 7: AIS-like output-boundary compensation for dendritic load."""
from __future__ import annotations

import hashlib

import numpy as np

from anttis_neuron.cable import cable_operator, simulate_cable
from anttis_neuron.output_boundary import (
    calibrate_threshold,
    driving_point_conductance,
    homeostatic_gain,
    rate_boundary,
    transfer_metrics,
)
from anttis_neuron.worlds import CableWorld, port_sensor_matrices
from experiments.gate5_adaptive_cable import _COUPLING, _DT, _LEAK
from experiments.gate6_many_worlds import trained_world_conductances


SCALES = np.asarray((0.50, 0.75, 1.00, 1.25, 1.50), dtype=float)
_BURN = 250
_TARGET_RATE = 0.25
_HOMEOSTATIC_PHASES = 12
_HOMEOSTATIC_ETA = 0.5
_GAIN_MIN = 0.2
_GAIN_MAX = 5.0
_DENDRITIC_TAPE_SEED = 6017
_COLLAPSE_LOW = 0.02
_COLLAPSE_HIGH = 0.98


def _digest_tape(tape: np.ndarray) -> str:
    data = np.ascontiguousarray(tape, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def _make_gate7_tapes(world: CableWorld) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic matched-distribution calibration/evaluation probes."""
    cal_rng = np.random.default_rng(
        np.random.SeedSequence([7017, world.seed, world.index])
    )
    eval_rng = np.random.default_rng(
        np.random.SeedSequence([8017, world.seed, world.index])
    )
    calibration = cal_rng.standard_normal((3000, 4))
    evaluation = eval_rng.standard_normal((5000, 4))
    return calibration, evaluation


def _world_operator(
    world: CableWorld,
    conductances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b, s = port_sensor_matrices(world)
    a = cable_operator(
        world.n_nodes,
        world.edges,
        conductances,
        dt=_DT,
        leak=_LEAK,
        coupling=_COUPLING,
    )
    return a, b, s


def _soma_voltage(
    world: CableWorld,
    conductances: np.ndarray,
    tape: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Return post-burn node-0 voltage for a scaled four-channel probe tape."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    a, b, s = _world_operator(world, conductances)
    _, states = simulate_cable(
        np.asarray(tape, dtype=float) * scale,
        a,
        b,
        s,
        burn=_BURN,
        return_states=True,
    )
    voltage = states[:, 0]
    if voltage.ndim != 1 or len(voltage) == 0 or not np.all(np.isfinite(voltage)):
        raise FloatingPointError("invalid soma voltage trace")
    return voltage


def _mean_rate(voltage: np.ndarray, *, gain: float, theta: float, beta: float) -> float:
    return float(np.mean(rate_boundary(voltage, gain=gain, theta=theta, beta=beta)))


def _adapt_homeostatic_gain(
    calibration_voltage: np.ndarray,
    *,
    theta: float,
    beta: float,
    target_rate: float,
) -> float:
    """Adapt gain using only the boundary's own mean rate error."""
    gain = 1.0
    for _ in range(_HOMEOSTATIC_PHASES):
        mean_rate = _mean_rate(
            calibration_voltage,
            gain=gain,
            theta=theta,
            beta=beta,
        )
        gain = homeostatic_gain(
            gain,
            mean_rate,
            target_rate=target_rate,
            eta=_HOMEOSTATIC_ETA,
            gain_min=_GAIN_MIN,
            gain_max=_GAIN_MAX,
        )
    return float(gain)


def _rate_curve(
    world: CableWorld,
    conductances: np.ndarray,
    evaluation_tape: np.ndarray,
    *,
    gain: float,
    theta: float,
    beta: float,
) -> np.ndarray:
    rates = []
    for scale in SCALES:
        voltage = _soma_voltage(
            world,
            conductances,
            evaluation_tape,
            scale=float(scale),
        )
        rates.append(_mean_rate(voltage, gain=gain, theta=theta, beta=beta))
    result = np.asarray(rates, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0) or np.any(result > 1.0):
        raise FloatingPointError("invalid Gate 7 transfer curve")
    return result


def _reference_parameters(
    reference_world: CableWorld,
    reference_g: np.ndarray,
) -> dict:
    """Calibrate the fixed output boundary only from reference-world calibration data."""
    calibration_tape, evaluation_tape = _make_gate7_tapes(reference_world)
    calibration_voltage = _soma_voltage(
        reference_world,
        reference_g,
        calibration_tape,
    )
    voltage_std = float(np.std(calibration_voltage))
    if not np.isfinite(voltage_std) or voltage_std <= 1e-12:
        raise FloatingPointError("reference soma-voltage standard deviation is invalid")
    beta = 2.0 / voltage_std
    theta = calibrate_threshold(
        calibration_voltage,
        gain=1.0,
        beta=beta,
        target_rate=_TARGET_RATE,
    )
    calibration_rate = _mean_rate(
        calibration_voltage,
        gain=1.0,
        theta=theta,
        beta=beta,
    )
    reference_load = driving_point_conductance(
        reference_world.n_nodes,
        reference_world.edges,
        reference_g,
        leak=_LEAK,
        coupling=_COUPLING,
        soma_node=0,
    )
    reference_curve = _rate_curve(
        reference_world,
        reference_g,
        evaluation_tape,
        gain=1.0,
        theta=theta,
        beta=beta,
    )
    dynamic_range = float(np.max(reference_curve) - np.min(reference_curve))
    if dynamic_range < 0.10:
        raise FloatingPointError(
            f"reference transfer curve is too flat: dynamic range={dynamic_range}"
        )
    return {
        "reference_world_index": int(reference_world.index),
        "target_rate": _TARGET_RATE,
        "beta": float(beta),
        "theta": float(theta),
        "reference_load": float(reference_load),
        "reference_curve": [float(value) for value in reference_curve],
        "reference_curve_dynamic_range": dynamic_range,
        "reference_unit_calibration_rate": float(calibration_rate),
        "reference_calibration_tape_digest": _digest_tape(calibration_tape),
        "reference_evaluation_tape_digest": _digest_tape(evaluation_tape),
    }


def _condition_result(
    rates: np.ndarray,
    reference_curve: np.ndarray,
    *,
    gain: float,
) -> dict:
    metrics = transfer_metrics(SCALES, rates, reference_curve, rheobase_rate=0.10)
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
    """Evaluate fixed, local-homeostatic, and load-oracle boundaries in one world."""
    trained = trained_world_conductances(world, tape_seed=_DENDRITIC_TAPE_SEED)
    conductances = trained["local"]
    calibration_tape, evaluation_tape = _make_gate7_tapes(world)
    calibration_voltage = _soma_voltage(world, conductances, calibration_tape)

    theta = float(reference["theta"])
    beta = float(reference["beta"])
    target_rate = float(reference["target_rate"])
    reference_load = float(reference["reference_load"])
    reference_curve = np.asarray(reference["reference_curve"], dtype=float)

    load = driving_point_conductance(
        world.n_nodes,
        world.edges,
        conductances,
        leak=_LEAK,
        coupling=_COUPLING,
        soma_node=0,
    )
    fixed_gain = 1.0
    homeostatic = _adapt_homeostatic_gain(
        calibration_voltage,
        theta=theta,
        beta=beta,
        target_rate=target_rate,
    )
    oracle_gain = float(load / reference_load)

    # The three conditions differ only at the output boundary. Their physical
    # input is the exact same deterministic evaluation tape and fixed cable.
    fixed_rates = _rate_curve(
        world,
        conductances,
        evaluation_tape,
        gain=fixed_gain,
        theta=theta,
        beta=beta,
    )
    homeostatic_rates = _rate_curve(
        world,
        conductances,
        evaluation_tape,
        gain=homeostatic,
        theta=theta,
        beta=beta,
    )
    oracle_rates = _rate_curve(
        world,
        conductances,
        evaluation_tape,
        gain=oracle_gain,
        theta=theta,
        beta=beta,
    )

    fixed = _condition_result(fixed_rates, reference_curve, gain=fixed_gain)
    local = _condition_result(homeostatic_rates, reference_curve, gain=homeostatic)
    oracle = _condition_result(oracle_rates, reference_curve, gain=oracle_gain)

    return {
        "world_index": int(world.index),
        "world_seed": int(world.seed),
        "topology_edges": [[int(i), int(j)] for i, j in world.edges],
        "port_nodes": [int(value) for value in world.port_nodes],
        "load": float(load),
        "reference_load": reference_load,
        "calibration_tape_digest": _digest_tape(calibration_tape),
        "evaluation_tape_digest": _digest_tape(evaluation_tape),
        "scales": [float(value) for value in SCALES],
        "fixed_gain_start": 1.0,
        "homeostatic_gain_start": 1.0,
        "oracle_gain": oracle_gain,
        "fixed": fixed,
        "homeostatic": local,
        "oracle": oracle,
        "homeostatic_delta": float(
            fixed["metrics"]["curve_rmse"] - local["metrics"]["curve_rmse"]
        ),
        "oracle_delta": float(
            fixed["metrics"]["curve_rmse"] - oracle["metrics"]["curve_rmse"]
        ),
    }
