"""Gate 0: Oja recovers a known leading covariance mode."""
from __future__ import annotations

import numpy as np

from anttis_neuron.oja import absolute_cosine, oja_fit, principal_eigenvector
from experiments.common import main


def run(seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    cov = q @ np.diag([9.0, 3.0, 1.5, 0.7, 0.3]) @ q.T
    samples = rng.multivariate_normal(np.zeros(5), cov, size=14000)
    learned = oja_fit(samples, lr=0.0015, epochs=3, seed=seed + 1)
    target = principal_eigenvector(cov)
    return {
        "gate": 0,
        "seed": seed,
        "oja_alignment": absolute_cosine(learned, target),
        "n_samples": int(samples.shape[0]),
    }


if __name__ == "__main__":
    main(run)
