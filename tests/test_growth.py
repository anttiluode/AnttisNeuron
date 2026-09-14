import math

import pytest

from anttis_neuron.growth import (
    grow_to_purity,
    modal_energy_purity,
    required_continuous_length,
)


def test_modal_energy_purity_matches_closed_form():
    purity = modal_energy_purity(
        target_amplitude=1.0,
        distractor_amplitude=2.0,
        target_decay=0.03,
        distractor_decay=0.11,
        length=7.0,
    )
    target_energy = math.exp(-2.0 * 0.03 * 7.0)
    distractor_energy = 4.0 * math.exp(-2.0 * 0.11 * 7.0)
    assert math.isclose(purity, target_energy / (target_energy + distractor_energy), abs_tol=1e-12)


def test_growth_stops_at_first_integer_length_meeting_threshold():
    result = grow_to_purity(
        target_amplitude=1.0,
        distractor_amplitude=1.0,
        target_decay=0.03,
        distractor_decay=0.11,
        purity_target=0.95,
        max_length=200,
    )
    predicted = required_continuous_length(
        target_amplitude=1.0,
        distractor_amplitude=1.0,
        target_decay=0.03,
        distractor_decay=0.11,
        purity_target=0.95,
    )
    assert result.reached
    assert result.length == math.ceil(predicted - 1e-12)
    assert result.purity >= 0.95
    if result.length > 0:
        assert result.history[result.length - 1] < 0.95


def test_larger_modal_gap_requires_less_growth():
    slow_gap = grow_to_purity(1.0, 1.0, 0.03, 0.07, 0.95, 300)
    medium_gap = grow_to_purity(1.0, 1.0, 0.03, 0.11, 0.95, 300)
    wide_gap = grow_to_purity(1.0, 1.0, 0.03, 0.19, 0.95, 300)
    assert slow_gap.reached and medium_gap.reached and wide_gap.reached
    assert slow_gap.length > medium_gap.length > wide_gap.length


def test_more_initial_contamination_requires_more_growth():
    low = grow_to_purity(1.0, 0.5, 0.03, 0.11, 0.95, 300)
    equal = grow_to_purity(1.0, 1.0, 0.03, 0.11, 0.95, 300)
    high = grow_to_purity(1.0, 2.0, 0.03, 0.11, 0.95, 300)
    assert low.reached and equal.reached and high.reached
    assert low.length < equal.length < high.length


def test_equal_persistence_cannot_create_purity_from_length():
    initial = modal_energy_purity(1.0, 1.0, 0.08, 0.08, 0.0)
    result = grow_to_purity(1.0, 1.0, 0.08, 0.08, 0.95, 200)
    assert not result.reached
    assert result.length == 200
    assert all(math.isclose(value, initial, abs_tol=1e-12) for value in result.history)
    with pytest.raises(ValueError, match="equal persistence"):
        required_continuous_length(1.0, 1.0, 0.08, 0.08, 0.95)
