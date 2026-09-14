"""Gate 3: differential decay increases relative purity of a slow physical mode."""
from __future__ import annotations

import numpy as np

from anttis_neuron.branch import decay_ratio
from experiments.common import main


def run(seed: int = 17) -> dict:
    del seed
    mu_slow = 0.03
    mu_fast = 0.21
    times = [0, 5, 10, 20]
    a = np.diag([np.exp(-mu_slow), np.exp(-mu_fast)])
    state = np.ones(2)
    simulated = {0: float(state[0] / state[1])}
    for t in range(1, max(times) + 1):
        state = a @ state
        if t in times:
            simulated[t] = float(state[0] / state[1])
    predicted = {t: decay_ratio(mu_slow, mu_fast, t) for t in times}
    errors = [abs(simulated[t] - predicted[t]) for t in times]
    return {
        "gate": 3,
        "seed": 17,
        "mu_slow": mu_slow,
        "mu_fast": mu_fast,
        "ratio_t5": simulated[5],
        "ratio_t10": simulated[10],
        "ratio_t20": simulated[20],
        "max_ratio_error": float(max(errors)),
    }


if __name__ == "__main__":
    main(run)
