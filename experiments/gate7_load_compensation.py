"""Gate 7: local output-boundary compensation for somatic dendritic load."""
from __future__ import annotations

import argparse
import math

import numpy as np

from anttis_neuron.output_boundary import (
    driving_point_conductance,
    homeostatic_gain,
    rate_boundary,
    steady_state_soma_voltage,
    transfer_metrics,
)
from anttis_neuron.worlds import CableWorld, generate_world
from experiments.common import emit
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


def run_world(
    world: CableWorld,
    *,
    reference: dict,
    dendritic_tape_seed: int = _DENDRITIC_TAPE_SEED,
) -> dict:
    """Evaluate fixed, local-homeostatic, and exact load-oracle boundaries."""
    trained = trained_world_conductances(world, tape_seed=dendritic_tape_seed)
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
        "dendritic_tape_seed": int(dendritic_tape_seed),
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


def _pearson(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b):
        raise ValueError("correlation vectors must have matching lengths")
    if len(values_a) < 2:
        return None
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("correlation vectors must be finite")
    if float(np.std(a)) <= 1e-15 or float(np.std(b)) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _optional_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "q25": None, "min": None}
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "q25": float(np.quantile(data, 0.25)),
        "min": float(np.min(data)),
    }


def _aggregate(worlds: list[dict]) -> dict:
    if not worlds:
        raise ValueError("Gate 7 aggregate requires at least one world")

    aggregate: dict[str, float | int | None] = {}
    nonreference = [world for world in worlds if world["world_index"] != 0]
    deltas = [float(world["homeostatic_delta"]) for world in nonreference]
    delta_summary = _optional_distribution(deltas)
    aggregate.update(
        {
            "homeostatic_delta_mean": delta_summary["mean"],
            "homeostatic_delta_median": delta_summary["median"],
            "homeostatic_delta_q25": delta_summary["q25"],
            "homeostatic_delta_min": delta_summary["min"],
            "homeostatic_beats_fixed_fraction": (
                None if not deltas else float(np.mean(np.asarray(deltas) > 0.0))
            ),
            "homeostatic_loses_to_fixed_count": int(np.sum(np.asarray(deltas) < 0.0)),
        }
    )

    for condition in ("fixed", "homeostatic", "oracle"):
        rmses = np.asarray(
            [world[condition]["metrics"]["curve_rmse"] for world in worlds],
            dtype=float,
        )
        unit_rates = np.asarray(
            [world[condition]["metrics"]["unit_rate"] for world in worlds],
            dtype=float,
        )
        fi_gains = np.asarray(
            [world[condition]["metrics"]["fi_gain"] for world in worlds],
            dtype=float,
        )
        rheobases = [
            world[condition]["metrics"]["rheobase"]
            for world in worlds
            if world[condition]["metrics"]["rheobase"] is not None
        ]
        aggregate.update(
            {
                f"{condition}_curve_rmse_mean": float(np.mean(rmses)),
                f"{condition}_curve_rmse_median": float(np.median(rmses)),
                f"{condition}_curve_rmse_q75": float(np.quantile(rmses, 0.75)),
                f"{condition}_unit_rate_mean": float(np.mean(unit_rates)),
                f"{condition}_unit_rate_std": float(np.std(unit_rates)),
                f"{condition}_fi_gain_std": float(np.std(fi_gains)),
                f"{condition}_rheobase_std": (
                    None if not rheobases else float(np.std(np.asarray(rheobases, dtype=float)))
                ),
                f"{condition}_rheobase_missing_count": int(len(worlds) - len(rheobases)),
                f"{condition}_collapse_count": int(
                    sum(bool(world[condition]["collapsed"]) for world in worlds)
                ),
            }
        )

    aggregate["oracle_curve_rmse_max"] = float(
        max(world["oracle"]["metrics"]["curve_rmse"] for world in worlds)
    )
    aggregate["oracle_numerical_error_max"] = float(
        max(world["oracle_numerical_error"] for world in worlds)
    )
    aggregate["load_ratio_min"] = float(min(world["load_ratio"] for world in worlds))
    aggregate["load_ratio_max"] = float(max(world["load_ratio"] for world in worlds))
    aggregate["load_ratio_vs_homeostatic_gain_correlation"] = _pearson(
        [float(world["load_ratio"]) for world in worlds],
        [float(world["homeostatic"]["gain"]) for world in worlds],
    )
    aggregate["homeostatic_gain_vs_oracle_gain_correlation"] = _pearson(
        [float(world["homeostatic"]["gain"]) for world in worlds],
        [float(world["oracle_gain"]) for world in worlds],
    )
    return aggregate


def run(seed: int = 17, n_worlds: int = 24) -> dict:
    """Run the predeclared Gate-7 suite without outcome-dependent tuning."""
    if not 1 <= n_worlds <= 24:
        raise ValueError("Gate 7 requires 1 <= n_worlds <= 24")

    master_seed = 1701 + seed
    dendritic_tape_seed = 6000 + seed
    reference_world = generate_world(0, master_seed=master_seed)
    reference_g = trained_world_conductances(
        reference_world,
        tape_seed=dendritic_tape_seed,
    )["local"]
    reference = _reference_parameters(reference_world, reference_g)

    world_results = [
        run_world(
            generate_world(index, master_seed=master_seed),
            reference=reference,
            dendritic_tape_seed=dendritic_tape_seed,
        )
        for index in range(n_worlds)
    ]

    return {
        "gate": 7,
        "seed": int(seed),
        "n_worlds": int(n_worlds),
        "currents": [float(value) for value in CURRENTS],
        "protocol": {
            "assay": "deterministic steady somatic current into frozen Gate-6 local-adapted dendritic load",
            "soma_node": 0,
            "target_rate": _TARGET_RATE,
            "reference_slope_coefficient": _REFERENCE_SLOPE_COEFFICIENT,
            "homeostatic_phases": _HOMEOSTATIC_PHASES,
            "homeostatic_eta": _HOMEOSTATIC_ETA,
            "gain_min": _GAIN_MIN,
            "gain_max": _GAIN_MAX,
            "dendritic_tape_seed": dendritic_tape_seed,
            "collapse_low": _COLLAPSE_LOW,
            "collapse_high": _COLLAPSE_HIGH,
            "oracle": "unclipped driving-point load ratio; hidden from local homeostat",
        },
        "reference": reference,
        "aggregate": _aggregate(world_results),
        "worlds": world_results,
        "interpretation": (
            "synthetic somatic-load compensation test: dendritic conductances are frozen before "
            "output adaptation; the local AIS-like homeostat reads only its own rate error and "
            "cannot read load, topology, conductances, spectral information, or curve error; "
            "the oracle is an exact engineering identity control, and no positive local result "
            "is required by CI"
        ),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worlds", type=int, default=24)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    emit(run(seed=args.seed, n_worlds=args.worlds), args.out)


if __name__ == "__main__":
    _main()
