import math
from pathlib import Path
import json

import numpy as np

from experiments.gate6b_feature_ranking import (
    POST_ADAPTATION_FEATURES,
    PRE_EXISTING_FEATURES,
    _tree_distances,
    run,
)


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    if isinstance(value, str) or value is None or isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def test_tree_distances_on_a_known_path():
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    distances = _tree_distances(4, edges)
    assert distances[0][3] == 3
    assert distances[1][3] == 2
    assert distances[2][2] == 0


def test_gate6b_six_world_smoke_is_deterministic_and_finite():
    a = run(seed=17, n_worlds=6)
    b = run(seed=17, n_worlds=6)
    assert a == b
    assert _finite(a)
    assert a["gate"] == "6B"
    assert a["seed"] == 17
    assert a["n_worlds"] == 6
    assert len(a["worlds"]) == 6


def test_gate6b_rejects_too_few_worlds_for_correlation():
    for n_worlds in (1, 2):
        try:
            run(seed=17, n_worlds=n_worlds)
        except ValueError:
            pass
        else:
            raise AssertionError("Gate 6B's correlation ranking must reject fewer than 3 worlds")


def test_gate6b_feature_groups_are_disjoint_and_complete():
    a = run(seed=17, n_worlds=6)
    pre = set(a["feature_groups"]["pre_existing"])
    post = set(a["feature_groups"]["post_adaptation"])
    assert pre == set(PRE_EXISTING_FEATURES)
    assert post == set(POST_ADAPTATION_FEATURES)
    assert pre.isdisjoint(post)


def test_gate6b_r2_values_are_bounded():
    a = run(seed=17, n_worlds=6)
    for block in a["single_feature_r2"].values():
        for value in block.values():
            assert 0.0 <= value <= 1.0 + 1e-9
    assert 0.0 <= a["best_pre_existing_pair_vs_delta_frozen"]["r2"] <= 1.0 + 1e-9


def test_gate6b_world_rows_are_within_expected_ranges():
    a = run(seed=17, n_worlds=3)
    for world in a["worlds"]:
        assert -1.0 <= world["delta_frozen"] <= 1.0
        assert -1.0 <= world["delta_shuffled"] <= 1.0
        assert 0.0 <= world["initial_alignment"] <= 1.0
        assert world["tree_depth"] >= 1
        assert world["spectral_gap"] >= 0.0
        assert world["conductance_cv"] >= 0.0
        assert world["basis_rotation_deg"] >= 0.0


def test_gate6b_mechanism_summary_sign_match_counts_are_bounded():
    a = run(seed=17, n_worlds=5)
    summary = a["mechanism_summary"]
    assert 0 <= summary["eigvec_sign_matches_delta_frozen"] <= 5
    assert 0 <= summary["eigval_sign_matches_delta_frozen"] <= 5
    assert summary["n_worlds_scored"] == 5


def test_gate6b_rejects_world_count_outside_predeclared_suite():
    for n_worlds in (0, 25):
        try:
            run(seed=17, n_worlds=n_worlds)
        except ValueError:
            pass
        else:
            raise AssertionError("Gate 6B must enforce 1 <= n_worlds <= 24")


def test_gate6b_frozen_receipt_reproduces_gate6_exactly():
    receipt = json.loads(Path("results/gate6b.json").read_text(encoding="utf-8"))
    assert receipt["n_worlds"] == 24
    reproduction = receipt["reproduces_gate6_receipt"]
    assert reproduction["max_mismatch_delta_frozen"] == 0.0
    assert reproduction["max_mismatch_delta_shuffled"] == 0.0


def test_gate6b_frozen_receipt_headline_numbers():
    receipt = json.loads(Path("results/gate6b.json").read_text(encoding="utf-8"))
    frozen_r2 = receipt["single_feature_r2"]["post_adaptation_vs_delta_frozen"]
    assert frozen_r2["eigvec_only_gain"] > frozen_r2["eigval_only_gain"]
    pre_r2 = receipt["single_feature_r2"]["pre_existing_vs_delta_frozen"]
    assert pre_r2["initial_alignment"] == max(pre_r2.values())
    summary = receipt["mechanism_summary"]
    assert summary["eigvec_sign_matches_delta_frozen"] > summary["eigval_sign_matches_delta_frozen"]
