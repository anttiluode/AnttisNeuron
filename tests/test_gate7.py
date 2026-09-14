import json
import math
from pathlib import Path

from experiments.gate7_growth_to_purity import run


def test_gate7_growth_matches_predeclared_purity_law():
    result = run(seed=17)
    assert result["gate"] == 7
    assert result["purity_target"] == 0.95
    assert result["seed"] == 17

    for case in result["separable_cases"]:
        assert case["reached"]
        assert case["growth_length"] == case["predicted_integer_length"]
        assert case["final_purity"] >= result["purity_target"]
        assert case["previous_purity"] < result["purity_target"] or case["growth_length"] == 0


def test_gate7_modal_gap_controls_required_length():
    result = run(seed=17)
    gap_cases = result["gap_sweep"]
    gaps = [case["modal_gap"] for case in gap_cases]
    lengths = [case["growth_length"] for case in gap_cases]
    scaled = [case["continuous_length"] * case["modal_gap"] for case in gap_cases]

    assert gaps == sorted(gaps)
    assert all(a > b for a, b in zip(lengths, lengths[1:]))
    assert max(scaled) - min(scaled) < 1e-12


def test_gate7_contamination_controls_required_length():
    result = run(seed=17)
    contamination_cases = result["contamination_sweep"]
    contamination = [case["distractor_amplitude"] for case in contamination_cases]
    lengths = [case["growth_length"] for case in contamination_cases]

    assert contamination == sorted(contamination)
    assert all(a < b for a, b in zip(lengths, lengths[1:]))


def test_gate7_equal_persistence_is_a_hard_negative_control():
    control = run(seed=17)["equal_persistence_control"]
    assert not control["reached"]
    assert control["growth_length"] == control["max_length"]
    assert math.isclose(control["initial_purity"], control["final_purity"], abs_tol=1e-12)
    assert math.isclose(control["purity_change"], 0.0, abs_tol=1e-12)


def test_gate7_frozen_receipt_matches_experiment_exactly():
    receipt = json.loads(Path("results/gate7.json").read_text(encoding="utf-8"))
    assert receipt == run(seed=17)
