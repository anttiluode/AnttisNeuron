"""Gate 5: can local structural plasticity align statistics with branched physical modes?"""
from __future__ import annotations

import hashlib

import numpy as np

from anttis_neuron.branch import spectral_radius
from anttis_neuron.cable import (
    adapt_conductances,
    cable_operator,
    default_ports_and_sensors,
    default_tree,
    edge_difference_energy,
    simulate_cable,
    visible_physical_modes,
)
from anttis_neuron.oja import absolute_cosine, oja_fit
from experiments.common import main


_DT = 0.08
_LEAK = 0.08
_COUPLING = 0.65
_ETA = 0.10
_G_MIN = 0.45
_G_MAX = 1.8


def _covariance(angle_degrees: float, spectrum: list[float]) -> np.ndarray:
    theta = np.deg2rad(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.eye(4, dtype=float)
    rotation[:2, :2] = np.array([[c, -s], [s, c]])
    return rotation @ np.diag(np.asarray(spectrum, dtype=float)) @ rotation.T


def _make_input_tapes(seed: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(seed)
    adaptation: list[np.ndarray] = []
    for index, angle in enumerate((12.0, 31.0, 53.0, 74.0)):
        spectrum = [2.6, 0.8, 1.2, 0.6] if index % 2 == 0 else [1.8, 0.7, 2.1, 0.5]
        adaptation.append(rng.multivariate_normal(np.zeros(4), _covariance(angle, spectrum), size=2500))
    heldout = [
        rng.multivariate_normal(
            np.zeros(4),
            _covariance(angle, [2.2, 0.9, 1.6, 0.55]),
            size=5000,
        )
        for angle in (22.0, 42.0, 63.0, 83.0, 103.0)
    ]
    return adaptation, heldout


def _tape_digest(adaptation: list[np.ndarray], heldout: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for tape in adaptation + heldout:
        contiguous = np.ascontiguousarray(tape, dtype=np.float64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _operator(conductances: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes, edges = default_tree()
    b, s = default_ports_and_sensors(n_nodes)
    a = cable_operator(
        n_nodes,
        edges,
        conductances,
        dt=_DT,
        leak=_LEAK,
        coupling=_COUPLING,
    )
    return a, b, s, edges


def _initial_q_target(first_tape: np.ndarray) -> float:
    g = np.ones(10, dtype=float)
    a, b, s, edges = _operator(g)
    _, states = simulate_cable(first_tape, a, b, s, return_states=True)
    target = float(np.median(edge_difference_energy(states, edges)))
    if target <= 0.0 or not np.isfinite(target):
        raise FloatingPointError("invalid edge-energy target")
    return target


def _train_structure(
    mode: str,
    adaptation_tapes: list[np.ndarray],
    *,
    q_target: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    conductances = np.ones(10, dtype=float)
    energy_history: list[np.ndarray] = []
    shuffle_rng = np.random.default_rng(seed + 1000)

    for tape in adaptation_tapes:
        a, b, s, edges = _operator(conductances)
        _, states = simulate_cable(tape, a, b, s, return_states=True)
        energy_history.append(edge_difference_energy(states, edges))
        if mode == "frozen":
            continue
        conductances = adapt_conductances(
            states,
            edges,
            conductances,
            eta=_ETA,
            q_target=q_target,
            g_min=_G_MIN,
            g_max=_G_MAX,
            mode=mode,
            rng=shuffle_rng if mode == "shuffled" else None,
        )

    return conductances, np.mean(np.asarray(energy_history), axis=0)


def _evaluate(conductances: np.ndarray, tapes: list[np.ndarray], *, oja_seed_base: int) -> dict:
    a, b, s, _ = _operator(conductances)
    # The score concerns persistent nontrivial structure, so use the three
    # slowest visible modes after excluding the uniform global leak mode.
    physical_modes = visible_physical_modes(a, s)[:3]
    if len(physical_modes) == 0:
        raise FloatingPointError("no visible nontrivial physical modes")

    alignments: list[float] = []
    state_rms: list[float] = []
    for index, tape in enumerate(tapes):
        observations, states = simulate_cable(tape, a, b, s, burn=250, return_states=True)
        rms = float(np.sqrt(np.mean(np.sum(observations * observations, axis=1))))
        if rms <= 0.0 or not np.isfinite(rms):
            raise FloatingPointError("invalid observation RMS")
        normalized = observations / rms
        learned = oja_fit(normalized, lr=0.001, epochs=3, seed=oja_seed_base + index)
        alignments.append(max(absolute_cosine(learned, mode) for mode in physical_modes))
        state_rms.append(float(np.sqrt(np.mean(np.sum(states * states, axis=1)))))

    eigenvalues = np.linalg.eigvalsh(a)
    slow_nonuniform = float(np.sort(eigenvalues)[-2])
    return {
        "alignments": [float(v) for v in alignments],
        "mean_alignment": float(np.mean(alignments)),
        "median_alignment": float(np.median(alignments)),
        "min_alignment": float(np.min(alignments)),
        "mean_state_rms": float(np.mean(state_rms)),
        "spectral_radius": spectral_radius(a),
        "slow_nonuniform_eigenvalue": slow_nonuniform,
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run(seed: int = 17) -> dict:
    adaptation_tapes, heldout_tapes = _make_input_tapes(seed)
    digest = _tape_digest(adaptation_tapes, heldout_tapes)
    q_target = _initial_q_target(adaptation_tapes[0])

    trained: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for mode in ("frozen", "local", "shuffled", "uniform"):
        trained[mode] = _train_structure(mode, adaptation_tapes, q_target=q_target, seed=seed)

    # Common Oja initialization per tape across every structural condition.
    heldout_eval = {
        mode: _evaluate(g, heldout_tapes, oja_seed_base=seed + 3000)
        for mode, (g, _) in trained.items()
    }
    adaptation_eval = {
        mode: _evaluate(g, adaptation_tapes, oja_seed_base=seed + 2000)
        for mode, (g, _) in trained.items()
    }

    frozen_g = trained["frozen"][0]
    adaptive_g, adaptive_energy = trained["local"]
    shuffled_g = trained["shuffled"][0]
    uniform_g = trained["uniform"][0]

    adaptive_mean = heldout_eval["local"]["mean_alignment"]
    frozen_mean = heldout_eval["frozen"]["mean_alignment"]
    shuffled_mean = heldout_eval["shuffled"]["mean_alignment"]

    return {
        "gate": 5,
        "seed": seed,
        "topology_nodes": 11,
        "topology_edges": 10,
        "q_target": q_target,
        "common_tape_digest": digest,
        "control_tape_digest": digest,
        "frozen_conductances": frozen_g.tolist(),
        "adaptive_conductances": adaptive_g.tolist(),
        "shuffled_conductances": shuffled_g.tolist(),
        "uniform_conductances": uniform_g.tolist(),
        "adaptive_conductance_cv": float(np.std(adaptive_g) / np.mean(adaptive_g)),
        "shuffled_conductance_cv": float(np.std(shuffled_g) / np.mean(shuffled_g)),
        "adaptive_edge_energy_conductance_correlation": _correlation(
            adaptive_energy,
            adaptive_g - frozen_g,
        ),
        "frozen_adaptation_mean_alignment": adaptation_eval["frozen"]["mean_alignment"],
        "adaptive_adaptation_mean_alignment": adaptation_eval["local"]["mean_alignment"],
        "shuffled_adaptation_mean_alignment": adaptation_eval["shuffled"]["mean_alignment"],
        "uniform_adaptation_mean_alignment": adaptation_eval["uniform"]["mean_alignment"],
        "frozen_heldout_alignments": heldout_eval["frozen"]["alignments"],
        "adaptive_heldout_alignments": heldout_eval["local"]["alignments"],
        "shuffled_heldout_alignments": heldout_eval["shuffled"]["alignments"],
        "uniform_heldout_alignments": heldout_eval["uniform"]["alignments"],
        "frozen_heldout_mean_alignment": frozen_mean,
        "adaptive_heldout_mean_alignment": adaptive_mean,
        "shuffled_heldout_mean_alignment": shuffled_mean,
        "uniform_heldout_mean_alignment": heldout_eval["uniform"]["mean_alignment"],
        "frozen_heldout_median_alignment": heldout_eval["frozen"]["median_alignment"],
        "adaptive_heldout_median_alignment": heldout_eval["local"]["median_alignment"],
        "shuffled_heldout_median_alignment": heldout_eval["shuffled"]["median_alignment"],
        "frozen_heldout_min_alignment": heldout_eval["frozen"]["min_alignment"],
        "adaptive_heldout_min_alignment": heldout_eval["local"]["min_alignment"],
        "shuffled_heldout_min_alignment": heldout_eval["shuffled"]["min_alignment"],
        "adaptive_minus_frozen": adaptive_mean - frozen_mean,
        "adaptive_minus_shuffled": adaptive_mean - shuffled_mean,
        "frozen_spectral_radius": heldout_eval["frozen"]["spectral_radius"],
        "adaptive_spectral_radius": heldout_eval["local"]["spectral_radius"],
        "shuffled_spectral_radius": heldout_eval["shuffled"]["spectral_radius"],
        "uniform_spectral_radius": heldout_eval["uniform"]["spectral_radius"],
        "frozen_slow_nonuniform_eigenvalue": heldout_eval["frozen"]["slow_nonuniform_eigenvalue"],
        "adaptive_slow_nonuniform_eigenvalue": heldout_eval["local"]["slow_nonuniform_eigenvalue"],
        "shuffled_slow_nonuniform_eigenvalue": heldout_eval["shuffled"]["slow_nonuniform_eigenvalue"],
        "frozen_mean_state_rms": heldout_eval["frozen"]["mean_state_rms"],
        "adaptive_mean_state_rms": heldout_eval["local"]["mean_state_rms"],
        "shuffled_mean_state_rms": heldout_eval["shuffled"]["mean_state_rms"],
        "interpretation": "alignment-blind local structural test; scientific deltas are reported, not required positive by CI",
    }


if __name__ == "__main__":
    main(run)
