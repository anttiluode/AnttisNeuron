"""Gate 2: nonlinear spectral branches expose variance-only class structure."""
from __future__ import annotations

import numpy as np

from anttis_neuron.model import accuracy_from_voltage, ridge_fit, static_features
from anttis_neuron.oja import absolute_cosine, oja_fit
from anttis_neuron.tasks import make_local_oja_streams, make_mode_energy_task
from experiments.common import main


def _fit_accuracy(train_features, y_train, test_features, y_test) -> float:
    w, b = ridge_fit(train_features, y_train, l2=1e-5)
    return accuracy_from_voltage(test_features @ w + b, y_test)


def run(seed: int = 17) -> dict:
    xtr, ytr, xte, yte, true_modes = make_mode_energy_task(6000, 4000, seed=seed)
    streams = make_local_oja_streams(true_modes, 9000, seed=seed + 101)
    learned = np.stack(
        [oja_fit(stream, lr=0.001, epochs=2, seed=seed + 200 + j) for j, stream in enumerate(streams)]
    )
    alignments = [absolute_cosine(learned[j], true_modes[j]) for j in range(len(streams))]

    raw_acc = _fit_accuracy(xtr, ytr, xte, yte)
    id_train = static_features(xtr, learned, "identity")
    id_test = static_features(xte, learned, "identity")
    id_acc = _fit_accuracy(id_train, ytr, id_test, yte)
    sq_train = static_features(xtr, learned, "square")
    sq_test = static_features(xte, learned, "square")
    sq_acc = _fit_accuracy(sq_train, ytr, sq_test, yte)

    return {
        "gate": 2,
        "seed": seed,
        "raw_linear_accuracy": raw_acc,
        "identity_branch_accuracy": id_acc,
        "square_accuracy": sq_acc,
        "filter_alignment_0": float(alignments[0]),
        "filter_alignment_1": float(alignments[1]),
        "mean_filter_alignment": float(np.mean(alignments)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
    }


if __name__ == "__main__":
    main(run)
