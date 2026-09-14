"""Gate 6: robustness of the frozen Gate 5 rule across deterministic worlds."""
from __future__ import annotations

import argparse
import hashlib

import numpy as np

from anttis_neuron.branch import spectral_radius
from anttis_neuron.cable import (
    adapt_conductances,
    cable_operator,
    edge_difference_energy,
    simulate_cable,
    visible_physical_modes,
)
from anttis_neuron.oja import absolute_cosine, oja_fit
from anttis_neuron.worlds import CableWorld, generate_world, port_sensor_matrices
from experiments.common import emit
from experiments.gate5_adaptive_cable import (
    _COUPLING,
    _DT,
    _ETA,
    _G_MAX,
    _G_MIN,
    _LEAK,
    _apply_signal,
    _covariance,
)


def _world_operator(world: CableWorld, conductances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _make_world_tapes(world: CableWorld, tape_seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    sequence = np.random.SeedSequence([tape_seed, world.seed, world.index])
    rng = np.random.default_rng(sequence)
    adaptation = [
        rng.multivariate_normal(
            np.zeros(4),
            _covariance(angle, list(spectrum)),
            size=2500,
        )
        for angle, spectrum in zip(world.adaptation_angles, world.adaptation_spectra)
    ]
    heldout = [
        rng.multivariate_normal(
            np.zeros(4),
            _covariance(angle, list(world.heldout_spectrum)),
            size=5000,
        )
        for angle in world.heldout_angles
    ]
    return adaptation, heldout


def _digest_tapes(adaptation: list[np.ndarray], heldout: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for tape in adaptation + heldout:
        contiguous = np.ascontiguousarray(tape, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _initial_q_target(world: CableWorld, first_tape: np.ndarray) -> float:
    g = np.ones(len(world.edges), dtype=float)
    a, b, s = _world_operator(world, g)
    _, states = simulate_cable(first_tape, a, b, s, return_states=True)
    target = float(np.median(edge_difference_energy(states, world.edges)))
    if target <= 0.0 or not np.isfinite(target):
        raise FloatingPointError("invalid Gate 6 edge-energy target")
    return target


def _train_conditions(
    world: CableWorld,
    adaptation_tapes: list[np.ndarray],
    *,
    q_target: float,
    seed: int,
) -> tuple[dict[str, np.ndarray], float]:
    n_edges = len(world.edges)
    local_g = np.ones(n_edges, dtype=float)
    shuffled_g = np.ones(n_edges, dtype=float)
    uniform_g = np.ones(n_edges, dtype=float)
    frozen_g = np.ones(n_edges, dtype=float)
    shuffle_rng = np.random.default_rng(np.random.SeedSequence([seed, world.seed, 1000]))
    multiset_errors: list[float] = []

    for tape in adaptation_tapes:
        a, b, s = _world_operator(world, local_g)
        _, states = simulate_cable(tape, a, b, s, return_states=True)
        energy = edge_difference_energy(states, world.edges)
        local_signal = energy / q_target - 1.0

        local_g = adapt_conductances(
            states,
            world.edges,
            local_g,
            eta=_ETA,
            q_target=q_target,
            g_min=_G_MIN,
            g_max=_G_MAX,
            mode="local",
        )

        shuffled_signal = shuffle_rng.permutation(local_signal)
        multiset_errors.append(
            float(np.max(np.abs(np.sort(local_signal) - np.sort(shuffled_signal))))
        )
        shuffled_g = _apply_signal(shuffled_g, shuffled_signal)

        uniform_signal = np.full_like(local_signal, float(np.mean(local_signal)))
        uniform_g = _apply_signal(uniform_g, uniform_signal)

    return {
        "frozen": frozen_g,
        "local": local_g,
        "shuffled": shuffled_g,
        "uniform": uniform_g,
    }, float(max(multiset_errors, default=0.0))


def _evaluate(
    world: CableWorld,
    conductances: np.ndarray,
    tapes: list[np.ndarray],
    *,
    oja_seed_base: int,
) -> dict:
    a, b, s = _world_operator(world, conductances)
    physical_modes = visible_physical_modes(a, s)[:3]
    if len(physical_modes) < 3:
        raise FloatingPointError("Gate 6 world lost three visible physical modes")

    alignments: list[float] = []
    for index, tape in enumerate(tapes):
        observations = simulate_cable(tape, a, b, s, burn=250)
        rms = float(np.sqrt(np.mean(np.sum(observations * observations, axis=1))))
        if rms <= 0.0 or not np.isfinite(rms):
            raise FloatingPointError("invalid Gate 6 observation RMS")
        learned = oja_fit(
            observations / rms,
            lr=0.001,
            epochs=3,
            seed=oja_seed_base + index,
        )
        alignments.append(max(absolute_cosine(learned, mode) for mode in physical_modes))

    return {
        "alignments": [float(v) for v in alignments],
        "mean_alignment": float(np.mean(alignments)),
        "min_alignment": float(np.min(alignments)),
        "spectral_radius": spectral_radius(a),
    }


def run_world(world: CableWorld, *, tape_seed: int) -> dict:
    adaptation_tapes, heldout_tapes = _make_world_tapes(world, tape_seed)
    digest = _digest_tapes(adaptation_tapes, heldout_tapes)
    q_target = _initial_q_target(world, adaptation_tapes[0])
    trained, multiset_error = _train_conditions(
        world,
        adaptation_tapes,
        q_target=q_target,
        seed=tape_seed,
    )

    oja_seed_base = int(np.random.SeedSequence([tape_seed, world.seed, 3000]).generate_state(1)[0])
    evaluated = {
        name: _evaluate(world, g, heldout_tapes, oja_seed_base=oja_seed_base)
        for name, g in trained.items()
    }

    frozen = evaluated["frozen"]
    local = evaluated["local"]
    shuffled = evaluated["shuffled"]
    uniform = evaluated["uniform"]

    return {
        "world_index": world.index,
        "world_seed": world.seed,
        "topology_edges": [[int(i), int(j)] for i, j in world.edges],
        "port_nodes": [int(v) for v in world.port_nodes],
        "sensor_nodes": [int(v) for v in world.sensor_nodes],
        "adaptation_angles": [float(v) for v in world.adaptation_angles],
        "heldout_angles": [float(v) for v in world.heldout_angles],
        "q_target": q_target,
        "common_tape_digest": digest,
        "control_tape_digest": digest,
        "control_update_multiset_max_error": multiset_error,
        "adaptive_conductance_cv": float(np.std(trained["local"]) / np.mean(trained["local"])),
        "frozen_heldout_mean_alignment": frozen["mean_alignment"],
        "adaptive_heldout_mean_alignment": local["mean_alignment"],
        "shuffled_heldout_mean_alignment": shuffled["mean_alignment"],
        "uniform_heldout_mean_alignment": uniform["mean_alignment"],
        "frozen_heldout_min_alignment": frozen["min_alignment"],
        "adaptive_heldout_min_alignment": local["min_alignment"],
        "shuffled_heldout_min_alignment": shuffled["min_alignment"],
        "adaptive_minus_frozen": local["mean_alignment"] - frozen["mean_alignment"],
        "adaptive_minus_shuffled": local["mean_alignment"] - shuffled["mean_alignment"],
        "frozen_spectral_radius": frozen["spectral_radius"],
        "adaptive_spectral_radius": local["spectral_radius"],
        "shuffled_spectral_radius": shuffled["spectral_radius"],
        "uniform_spectral_radius": uniform["spectral_radius"],
    }


def _aggregate(worlds: list[dict]) -> dict:
    delta_frozen = np.asarray([w["adaptive_minus_frozen"] for w in worlds], dtype=float)
    delta_shuffled = np.asarray([w["adaptive_minus_shuffled"] for w in worlds], dtype=float)

    def mean_of(key: str) -> float:
        return float(np.mean([w[key] for w in worlds]))

    return {
        "adaptive_minus_frozen_mean": float(np.mean(delta_frozen)),
        "adaptive_minus_frozen_median": float(np.median(delta_frozen)),
        "adaptive_minus_frozen_q25": float(np.quantile(delta_frozen, 0.25)),
        "adaptive_minus_frozen_min": float(np.min(delta_frozen)),
        "adaptive_minus_shuffled_mean": float(np.mean(delta_shuffled)),
        "adaptive_minus_shuffled_median": float(np.median(delta_shuffled)),
        "adaptive_minus_shuffled_q25": float(np.quantile(delta_shuffled, 0.25)),
        "adaptive_minus_shuffled_min": float(np.min(delta_shuffled)),
        "adaptive_beats_frozen_fraction": float(np.mean(delta_frozen > 0.0)),
        "adaptive_beats_shuffled_fraction": float(np.mean(delta_shuffled > 0.0)),
        "adaptive_loses_to_frozen_count": int(np.sum(delta_frozen < 0.0)),
        "adaptive_loses_to_shuffled_count": int(np.sum(delta_shuffled < 0.0)),
        "frozen_mean_alignment": mean_of("frozen_heldout_mean_alignment"),
        "adaptive_mean_alignment": mean_of("adaptive_heldout_mean_alignment"),
        "shuffled_mean_alignment": mean_of("shuffled_heldout_mean_alignment"),
        "uniform_mean_alignment": mean_of("uniform_heldout_mean_alignment"),
    }


def run(seed: int = 17, n_worlds: int = 24) -> dict:
    if not 1 <= n_worlds <= 24:
        raise ValueError("Gate 6 requires 1 <= n_worlds <= 24")
    master_seed = 1701 + seed
    world_results = [
        run_world(
            generate_world(index, master_seed=master_seed),
            tape_seed=seed + 6000,
        )
        for index in range(n_worlds)
    ]
    return {
        "gate": 6,
        "seed": seed,
        "n_worlds": n_worlds,
        "rule": {
            "dt": _DT,
            "leak": _LEAK,
            "coupling": _COUPLING,
            "eta": _ETA,
            "g_min": _G_MIN,
            "g_max": _G_MAX,
            "oja_lr": 0.001,
            "oja_epochs": 3,
            "burn": 250,
        },
        "aggregate": _aggregate(world_results),
        "worlds": world_results,
        "interpretation": (
            "predeclared many-world robustness test of the frozen Gate 5 local rule; "
            "negative worlds are retained and no positive delta is required by CI"
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
