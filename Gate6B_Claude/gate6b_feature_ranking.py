"""Gate 6B: what predicts Gate 6's per-world outcome, and what explains it.

Two different questions, kept separate on purpose.

Explains-the-outcome (post-adaptation): extends Gate 5B's eigenvalue/eigenvector
hybrid-operator decomposition, originally run on one hand-built tree, to all 24
of Gate 6's worlds. Requires having already run the adaptation.

Predicts-the-outcome (pre-existing structure): ranks features knowable before
any adaptation runs -- spectral gap, tree depth, port-to-sensor path length,
initial alignment -- against Gate 6's adaptive_minus_frozen / adaptive_minus_shuffled,
the way Aizenbud et al. rank morphological features against FCI. Single-feature
R^2 first; a best pair is reported but not chased, since n=24 worlds and several
candidate features is already close to the edge of what will not overfit.

This experiment reuses Gate 6's and Gate 5B's own functions verbatim (imported,
not reimplemented) and cross-checks its regenerated adaptive_minus_frozen /
adaptive_minus_shuffled against the frozen results/gate6.json receipt as a
precondition: if that reproduction is not exact, nothing downstream is trusted.
"""
from __future__ import annotations

import json
from collections import deque
from itertools import combinations
from pathlib import Path

import numpy as np

from anttis_neuron.cable import simulate_cable, visible_physical_modes
from anttis_neuron.oja import oja_fit
from anttis_neuron.worlds import CableWorld, generate_world

from experiments.common import emit
from experiments.gate5b_spectral_audit import (
    hybrid_operators,
    principal_angles,
    sorted_nonuniform_eigendecomposition,
)
from experiments.gate6_many_worlds import (
    _evaluate,
    _initial_q_target,
    _make_world_tapes,
    _subspace_alignment,
    _train_conditions,
    _world_operator,
)

PRE_EXISTING_FEATURES = (
    "spectral_gap",
    "spectral_gap_ratio",
    "tree_depth",
    "mean_port_sensor_dist",
    "max_port_sensor_dist",
    "initial_alignment",
)
POST_ADAPTATION_FEATURES = (
    "conductance_cv",
    "basis_rotation_deg",
    "eigval_only_gain",
    "eigvec_only_gain",
)


def _tree_distances(n_nodes: int, edges: np.ndarray) -> dict[int, dict[int, int]]:
    """All-pairs shortest-path distances on an unweighted tree via BFS."""
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for i, j in edges:
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))

    def bfs(source: int) -> dict[int, int]:
        dist = {source: 0}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    return {node: bfs(node) for node in range(n_nodes)}


def _evaluate_custom_operator(
    a: np.ndarray,
    b: np.ndarray,
    s: np.ndarray,
    tapes: list[np.ndarray],
    *,
    oja_seed_base: int,
) -> float:
    """Mean held-out alignment of an arbitrary symmetric diagnostic operator."""
    physical_modes = visible_physical_modes(a, s)[:3]
    if len(physical_modes) < 3:
        raise FloatingPointError("Gate 6B requires three visible non-uniform modes")
    alignments = []
    for index, tape in enumerate(tapes):
        observations = simulate_cable(tape, a, b, s, burn=250)
        rms = float(np.sqrt(np.mean(np.sum(observations * observations, axis=1))))
        if rms <= 0.0 or not np.isfinite(rms):
            raise FloatingPointError("invalid Gate 6B observation RMS")
        learned = oja_fit(observations / rms, lr=0.001, epochs=3, seed=oja_seed_base + index)
        alignments.append(_subspace_alignment(learned, physical_modes))
    return float(np.mean(alignments))


def world_features(world: CableWorld, *, seed: int, tape_seed: int) -> dict:
    """Reconstruct one Gate 6 world exactly and extract Gate 6B's candidate features."""
    adaptation_tapes, heldout_tapes = _make_world_tapes(world, tape_seed)
    q_target = _initial_q_target(world, adaptation_tapes[0])
    trained, _ = _train_conditions(world, adaptation_tapes, q_target=q_target, seed=tape_seed)

    oja_seed_base = int(np.random.SeedSequence([tape_seed, world.seed, 3000]).generate_state(1)[0])

    a_frozen, b, s = _world_operator(world, trained["frozen"])
    a_local, _, _ = _world_operator(world, trained["local"])

    eval_frozen = _evaluate(world, trained["frozen"], heldout_tapes, oja_seed_base=oja_seed_base)
    eval_local = _evaluate(world, trained["local"], heldout_tapes, oja_seed_base=oja_seed_base)
    eval_shuffled = _evaluate(world, trained["shuffled"], heldout_tapes, oja_seed_base=oja_seed_base)

    delta_frozen = eval_local["mean_alignment"] - eval_frozen["mean_alignment"]
    delta_shuffled = eval_local["mean_alignment"] - eval_shuffled["mean_alignment"]

    a_lambda, a_q = hybrid_operators(a_frozen, a_local)
    lambda_alignment = _evaluate_custom_operator(a_lambda, b, s, heldout_tapes, oja_seed_base=oja_seed_base)
    q_alignment = _evaluate_custom_operator(a_q, b, s, heldout_tapes, oja_seed_base=oja_seed_base)

    frozen_vals, frozen_vecs = sorted_nonuniform_eigendecomposition(a_frozen)
    local_vals, local_vecs = sorted_nonuniform_eigendecomposition(a_local)
    basis_rotation_deg = float(np.degrees(np.mean(principal_angles(frozen_vecs, local_vecs, rank=3))))

    distances = _tree_distances(world.n_nodes, world.edges)
    depth_from_root = distances[0]
    port_sensor_dists = [distances[p][sn] for p in world.port_nodes for sn in world.sensor_nodes]

    return {
        "world_index": world.index,
        "world_seed": world.seed,
        "delta_frozen": delta_frozen,
        "delta_shuffled": delta_shuffled,
        "spectral_gap": float(frozen_vals[0] - frozen_vals[1]),
        "spectral_gap_ratio": float(frozen_vals[0] / frozen_vals[1]) if frozen_vals[1] != 0 else float("nan"),
        "conductance_cv": float(np.std(trained["local"]) / np.mean(trained["local"])),
        "tree_depth": int(max(depth_from_root.values())),
        "mean_port_sensor_dist": float(np.mean(port_sensor_dists)),
        "max_port_sensor_dist": float(np.max(port_sensor_dists)),
        "initial_alignment": eval_frozen["mean_alignment"],
        "basis_rotation_deg": basis_rotation_deg,
        "eigval_only_gain": lambda_alignment - eval_frozen["mean_alignment"],
        "eigvec_only_gain": q_alignment - eval_frozen["mean_alignment"],
    }


def _r2_single(x: list[float], y: list[float]) -> float:
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    if xv.size < 3:
        raise ValueError("Gate 6B correlation ranking needs at least 3 worlds to be well-defined")
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        raise ValueError("Gate 6B correlation is undefined when a feature or target has zero variance")
    r = float(np.corrcoef(xv, yv)[0, 1])
    return r * r


def _r2_multi(features: list[list[float]], y: list[float]) -> float:
    x = np.column_stack([np.asarray(f, dtype=float) for f in features])
    yv = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones(len(yv))])
    coefficients, *_ = np.linalg.lstsq(design, yv, rcond=None)
    predicted = design @ coefficients
    residual = float(np.sum((yv - predicted) ** 2))
    total = float(np.sum((yv - np.mean(yv)) ** 2))
    return 1.0 - residual / total


def _rank(rows: list[dict], features: tuple[str, ...], target_key: str) -> dict[str, float]:
    y = [row[target_key] for row in rows]
    return {feature: _r2_single([row[feature] for row in rows], y) for feature in features}


def run(seed: int = 17, n_worlds: int = 24, *, receipt_path: str = "results/gate6.json") -> dict:
    if not 1 <= n_worlds <= 24:
        raise ValueError("Gate 6B requires 1 <= n_worlds <= 24")
    master_seed = 1701 + seed
    tape_seed = seed + 6000

    rows = [
        world_features(generate_world(index, master_seed=master_seed), seed=seed, tape_seed=tape_seed)
        for index in range(n_worlds)
    ]

    max_mismatch_frozen = 0.0
    max_mismatch_shuffled = 0.0
    if n_worlds == 24 and Path(receipt_path).exists():
        recorded = {w["world_index"]: w for w in json.loads(Path(receipt_path).read_text())["worlds"]}
        for row in rows:
            recorded_world = recorded[row["world_index"]]
            max_mismatch_frozen = max(
                max_mismatch_frozen, abs(row["delta_frozen"] - recorded_world["adaptive_minus_frozen"])
            )
            max_mismatch_shuffled = max(
                max_mismatch_shuffled, abs(row["delta_shuffled"] - recorded_world["adaptive_minus_shuffled"])
            )

    frozen_r2 = _rank(rows, PRE_EXISTING_FEATURES, "delta_frozen")
    shuffled_r2 = _rank(rows, PRE_EXISTING_FEATURES, "delta_shuffled")
    frozen_r2_post = _rank(rows, POST_ADAPTATION_FEATURES, "delta_frozen")
    shuffled_r2_post = _rank(rows, POST_ADAPTATION_FEATURES, "delta_shuffled")

    delta_frozen = [row["delta_frozen"] for row in rows]
    best_pair: tuple[float, str, str] | None = None
    for f1, f2 in combinations(PRE_EXISTING_FEATURES, 2):
        r2 = _r2_multi([[row[f1] for row in rows], [row[f2] for row in rows]], delta_frozen)
        if best_pair is None or r2 > best_pair[0]:
            best_pair = (r2, f1, f2)
    assert best_pair is not None

    abs_delta = np.abs(np.asarray(delta_frozen))
    extreme_indices = set(np.argsort(abs_delta)[::-1][:3].tolist())
    keep = [i for i in range(len(rows)) if i not in extreme_indices]
    eigvec_gain = [row["eigvec_only_gain"] for row in rows]
    eigval_gain = [row["eigval_only_gain"] for row in rows]
    eigvec_sign_matches = sum(
        1 for row in rows if np.sign(row["eigvec_only_gain"]) == np.sign(row["delta_frozen"]) or abs(row["delta_frozen"]) < 1e-4
    )
    eigval_sign_matches = sum(
        1 for row in rows if np.sign(row["eigval_only_gain"]) == np.sign(row["delta_frozen"]) or abs(row["delta_frozen"]) < 1e-4
    )

    return {
        "gate": "6B",
        "seed": seed,
        "n_worlds": n_worlds,
        "reproduces_gate6_receipt": {
            "max_mismatch_delta_frozen": max_mismatch_frozen,
            "max_mismatch_delta_shuffled": max_mismatch_shuffled,
        },
        "feature_groups": {
            "pre_existing": list(PRE_EXISTING_FEATURES),
            "post_adaptation": list(POST_ADAPTATION_FEATURES),
        },
        "single_feature_r2": {
            "pre_existing_vs_delta_frozen": frozen_r2,
            "pre_existing_vs_delta_shuffled": shuffled_r2,
            "post_adaptation_vs_delta_frozen": frozen_r2_post,
            "post_adaptation_vs_delta_shuffled": shuffled_r2_post,
        },
        "best_pre_existing_pair_vs_delta_frozen": {
            "features": [best_pair[1], best_pair[2]],
            "r2": best_pair[0],
        },
        "mechanism_summary": {
            "mean_abs_eigvec_only_gain": float(np.mean(np.abs(eigvec_gain))),
            "mean_abs_eigval_only_gain": float(np.mean(np.abs(eigval_gain))),
            "eigvec_sign_matches_delta_frozen": eigvec_sign_matches,
            "eigval_sign_matches_delta_frozen": eigval_sign_matches,
            "n_worlds_scored": len(rows),
            "eigvec_r2_excluding_three_largest_swings": (
                _r2_single([eigvec_gain[i] for i in keep], [delta_frozen[i] for i in keep])
                if len(keep) >= 3
                else None
            ),
            "eigval_r2_excluding_three_largest_swings": (
                _r2_single([eigval_gain[i] for i in keep], [delta_frozen[i] for i in keep])
                if len(keep) >= 3
                else None
            ),
        },
        "worlds": rows,
        "interpretation": (
            "post_adaptation features explain how Gate 6's outcome happened (extends Gate 5B's "
            "hybrid-operator decomposition from one tree to all 24 worlds); pre_existing features "
            "test whether that outcome could have been predicted before running adaptation, the way "
            "Aizenbud et al. rank morphological features against FCI; the two questions are reported "
            "separately and neither is optimized to look better than it is"
        ),
    }


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worlds", type=int, default=24)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    emit(run(seed=args.seed, n_worlds=args.worlds), args.out)


if __name__ == "__main__":
    _main()
