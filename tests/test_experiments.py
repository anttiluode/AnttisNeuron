import math

from experiments.gate0_oja import run as run0
from experiments.gate1_identities import run as run1
from experiments.gate2_mode_energy import run as run2
from experiments.gate3_temporal_modes import run as run3
from experiments.gate4_alignment import run as run4
from experiments.gate5_adaptive_cable import run as run5


def _all_finite(value):
    if isinstance(value, dict):
        return all(_all_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(v) for v in value)
    if isinstance(value, bool) or isinstance(value, str) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def test_all_gates_are_finite_and_deterministic():
    for gate in (run0, run1, run2, run3, run4, run5):
        a = gate(seed=17)
        b = gate(seed=17)
        assert a == b
        assert _all_finite(a)


def test_gate0_recovers_principal_mode():
    assert run0(seed=17)["oja_alignment"] >= 0.98


def test_gate1_exact_identities_are_machine_precision():
    result = run1(seed=17)
    assert result["linear_collapse_max_abs_error"] < 1e-11
    assert result["two_layer_max_abs_error"] < 1e-11
    assert result["quadratic_max_abs_error"] < 1e-11


def test_gate2_requires_branch_nonlinearity():
    result = run2(seed=17)
    assert result["square_accuracy"] > 0.90
    assert result["square_accuracy"] > result["raw_linear_accuracy"] + 0.30
    assert result["square_accuracy"] > result["identity_branch_accuracy"] + 0.30
    assert result["mean_filter_alignment"] > 0.98


def test_gate3_matches_differential_decay_formula():
    result = run3(seed=17)
    assert result["max_ratio_error"] < 1e-12
    assert result["ratio_t20"] > result["ratio_t5"] > 1.0


def test_gate4_reports_controls_without_requiring_positive_result():
    result = run4(seed=17)
    assert 0.0 <= result["fixed_alignment"] <= 1.0
    assert 0.0 <= result["coupled_alignment"] <= 1.0
    assert 0.0 <= result["no_memory_alignment"] <= 1.0
    assert -1.0 <= result["delta_alignment"] <= 1.0


def test_gate5_reports_controls_without_requiring_positive_result():
    result = run5(seed=17)
    assert result["gate"] == 5
    assert result["seed"] == 17
    for key in (
        "frozen_heldout_mean_alignment",
        "adaptive_heldout_mean_alignment",
        "shuffled_heldout_mean_alignment",
        "uniform_heldout_mean_alignment",
    ):
        assert 0.0 <= result[key] <= 1.0
    assert -1.0 <= result["adaptive_minus_frozen"] <= 1.0
    assert -1.0 <= result["adaptive_minus_shuffled"] <= 1.0
    assert len(result["adaptive_conductances"]) == 10
    assert result["adaptive_spectral_radius"] < 1.0
    assert result["common_tape_digest"] == result["control_tape_digest"]
    assert result["control_update_multiset_max_error"] < 1e-12


def test_gate3_receipt_preserves_requested_seed():
    assert run3(seed=41)["seed"] == 41
