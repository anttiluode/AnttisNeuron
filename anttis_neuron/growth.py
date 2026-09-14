"""Minimal modal-growth model used by Gate 7.

The model isolates one question: if two physical modes attenuate at different
rates per unit path, can adding path length act as a purification budget?
It does not model a growth signal, synaptogenesis, ion channels, or biology.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GrowthResult:
    """Result of adding one unit of path at a time until purity is reached."""

    reached: bool
    length: int
    purity: float
    history: tuple[float, ...]


def _validate(
    target_amplitude: float,
    distractor_amplitude: float,
    target_decay: float,
    distractor_decay: float,
) -> None:
    values = (target_amplitude, distractor_amplitude, target_decay, distractor_decay)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("modal parameters must be finite")
    if target_decay < 0.0 or distractor_decay < 0.0:
        raise ValueError("decay rates must be non-negative")
    if target_amplitude == 0.0 and distractor_amplitude == 0.0:
        raise ValueError("at least one mode amplitude must be nonzero")


def modal_energy_purity(
    target_amplitude: float,
    distractor_amplitude: float,
    target_decay: float,
    distractor_decay: float,
    length: float,
) -> float:
    """Fraction of surviving modal energy carried by the target mode."""
    _validate(target_amplitude, distractor_amplitude, target_decay, distractor_decay)
    if not math.isfinite(length) or length < 0.0:
        raise ValueError("length must be finite and non-negative")

    target_energy = target_amplitude * target_amplitude * math.exp(-2.0 * target_decay * length)
    distractor_energy = distractor_amplitude * distractor_amplitude * math.exp(
        -2.0 * distractor_decay * length
    )
    total = target_energy + distractor_energy
    if total <= 0.0:
        raise FloatingPointError("modal energy underflowed to zero")
    return target_energy / total


def required_continuous_length(
    target_amplitude: float,
    distractor_amplitude: float,
    target_decay: float,
    distractor_decay: float,
    purity_target: float,
) -> float:
    """Closed-form path length required to reach an energy-purity target."""
    _validate(target_amplitude, distractor_amplitude, target_decay, distractor_decay)
    if not math.isfinite(purity_target) or not 0.0 < purity_target < 1.0:
        raise ValueError("purity_target must lie strictly between zero and one")

    initial = modal_energy_purity(
        target_amplitude,
        distractor_amplitude,
        target_decay,
        distractor_decay,
        0.0,
    )
    if initial >= purity_target:
        return 0.0
    if target_amplitude == 0.0:
        raise ValueError("zero target amplitude cannot be purified by passive growth")

    gap = distractor_decay - target_decay
    if abs(gap) <= 1e-15:
        raise ValueError("equal persistence cannot increase modal purity")
    if gap < 0.0:
        raise ValueError("target mode must be more persistent than distractor")

    energy_ratio = (distractor_amplitude * distractor_amplitude) / (
        target_amplitude * target_amplitude
    )
    odds_target = purity_target / (1.0 - purity_target)
    return math.log(odds_target * energy_ratio) / (2.0 * gap)


def grow_to_purity(
    target_amplitude: float,
    distractor_amplitude: float,
    target_decay: float,
    distractor_decay: float,
    purity_target: float,
    max_length: int,
) -> GrowthResult:
    """Add integer path units until the requested purity is first reached."""
    _validate(target_amplitude, distractor_amplitude, target_decay, distractor_decay)
    if not math.isfinite(purity_target) or not 0.0 < purity_target < 1.0:
        raise ValueError("purity_target must lie strictly between zero and one")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 0:
        raise ValueError("max_length must be a non-negative integer")

    history: list[float] = []
    for length in range(max_length + 1):
        purity = modal_energy_purity(
            target_amplitude,
            distractor_amplitude,
            target_decay,
            distractor_decay,
            float(length),
        )
        history.append(purity)
        if purity >= purity_target:
            return GrowthResult(True, length, purity, tuple(history))

    return GrowthResult(False, max_length, history[-1], tuple(history))
