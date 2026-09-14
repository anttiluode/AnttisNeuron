"""Gate 7: treat added physical path as a modal-purification budget.

This is deliberately an existence test, not a biological growth rule.  The target
mode is assumed to be specified (the analogue of an inherited/developmental
program), and the gate asks only whether differential modal attenuation makes the
required path length predictable.
"""
from __future__ import annotations

import math

from anttis_neuron.growth import (
    grow_to_purity,
    modal_energy_purity,
    required_continuous_length,
)
from experiments.common import main


_PURITY_TARGET = 0.95
_TARGET_DECAY = 0.03
_MAX_LENGTH = 300


def _separable_case(
    *,
    label: str,
    target_amplitude: float,
    distractor_amplitude: float,
    target_decay: float,
    distractor_decay: float,
) -> dict:
    continuous = required_continuous_length(
        target_amplitude,
        distractor_amplitude,
        target_decay,
        distractor_decay,
        _PURITY_TARGET,
    )
    result = grow_to_purity(
        target_amplitude,
        distractor_amplitude,
        target_decay,
        distractor_decay,
        _PURITY_TARGET,
        _MAX_LENGTH,
    )
    predicted_integer = int(math.ceil(continuous - 1e-12))
    previous = result.history[result.length - 1] if result.length > 0 else result.history[0]
    return {
        "label": label,
        "target_amplitude": float(target_amplitude),
        "distractor_amplitude": float(distractor_amplitude),
        "target_decay": float(target_decay),
        "distractor_decay": float(distractor_decay),
        "modal_gap": float(distractor_decay - target_decay),
        "initial_purity": float(result.history[0]),
        "continuous_length": float(continuous),
        "predicted_integer_length": predicted_integer,
        "growth_length": int(result.length),
        "previous_purity": float(previous),
        "final_purity": float(result.purity),
        "reached": bool(result.reached),
    }


def run(seed: int = 17) -> dict:
    # Algebraic/deterministic gate; seed is retained for the common receipt interface.
    gap_sweep = [
        _separable_case(
            label=f"gap_{gap:.2f}",
            target_amplitude=1.0,
            distractor_amplitude=1.0,
            target_decay=_TARGET_DECAY,
            distractor_decay=_TARGET_DECAY + gap,
        )
        for gap in (0.02, 0.04, 0.08, 0.16)
    ]

    contamination_sweep = [
        _separable_case(
            label=f"contamination_{amplitude:.1f}",
            target_amplitude=1.0,
            distractor_amplitude=amplitude,
            target_decay=_TARGET_DECAY,
            distractor_decay=_TARGET_DECAY + 0.08,
        )
        for amplitude in (0.5, 1.0, 2.0)
    ]

    equal = grow_to_purity(
        1.0,
        1.0,
        0.08,
        0.08,
        _PURITY_TARGET,
        _MAX_LENGTH,
    )
    equal_control = {
        "target_decay": 0.08,
        "distractor_decay": 0.08,
        "modal_gap": 0.0,
        "max_length": _MAX_LENGTH,
        "growth_length": int(equal.length),
        "initial_purity": float(equal.history[0]),
        "final_purity": float(equal.purity),
        "purity_change": float(equal.purity - equal.history[0]),
        "reached": bool(equal.reached),
    }

    separable = gap_sweep + contamination_sweep
    exact_match_fraction = sum(
        case["growth_length"] == case["predicted_integer_length"] for case in separable
    ) / len(separable)

    return {
        "gate": 7,
        "seed": seed,
        "name": "growth_to_purity",
        "purity_definition": "target surviving modal energy / total surviving modal energy",
        "purity_target": _PURITY_TARGET,
        "path_unit": "dimensionless serial attenuation unit",
        "growth_rule": "add one path unit until oracle target-mode purity first reaches threshold",
        "gap_sweep": gap_sweep,
        "contamination_sweep": contamination_sweep,
        "separable_cases": separable,
        "equal_persistence_control": equal_control,
        "aggregate": {
            "n_separable_cases": len(separable),
            "prediction_exact_match_fraction": float(exact_match_fraction),
            "shortest_growth_length": int(min(case["growth_length"] for case in separable)),
            "longest_growth_length": int(max(case["growth_length"] for case in separable)),
            "equal_persistence_reached": bool(equal.reached),
        },
        "law": (
            "L* = log((p/(1-p)) * (a_d^2/a_t^2)) / "
            "(2 * (mu_d-mu_t)); integer growth stops at ceil(L*)"
        ),
        "interpretation": (
            "constructive Gate-3 extension: differential modal decay can be exchanged for "
            "physical path depth, so branches facing smaller modal gaps or heavier initial "
            "contamination require more growth; equal-persistence modes are the hard negative "
            "control and cannot be purified by length alone. The target mode and purity signal "
            "are oracle/developmental inputs here, so this gate does not claim a biological "
            "growth sensor or explain how evolution specifies the target."
        ),
    }


if __name__ == "__main__":
    main(run)
