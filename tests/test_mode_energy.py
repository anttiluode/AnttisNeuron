import numpy as np
from anttis_neuron.model import accuracy_from_voltage, ridge_fit, static_features
from anttis_neuron.tasks import make_mode_energy_task


def test_mode_energy_task_is_reproducible():
    a = make_mode_energy_task(200, 100, seed=9)
    b = make_mode_energy_task(200, 100, seed=9)
    for left, right in zip(a[:4], b[:4]):
        assert np.array_equal(left, right)
    assert np.array_equal(a[4], b[4])


def test_mode_energy_classes_remain_zero_mean():
    xtr, ytr, _, _, _ = make_mode_energy_task(20000, 100, seed=2)
    assert np.linalg.norm(xtr[ytr == 0].mean(axis=0)) < 0.08
    assert np.linalg.norm(xtr[ytr == 1].mean(axis=0)) < 0.08


def test_square_true_modes_make_task_readable():
    xtr, ytr, xte, yte, modes = make_mode_energy_task(3000, 1500, seed=5)
    phi_tr = static_features(xtr, modes, "square")
    phi_te = static_features(xte, modes, "square")
    w, b = ridge_fit(phi_tr, ytr)
    assert accuracy_from_voltage(phi_te @ w + b, yte) > 0.90


def test_raw_linear_readout_cannot_solve_variance_only_task():
    xtr, ytr, xte, yte, _ = make_mode_energy_task(5000, 3000, seed=12)
    w, b = ridge_fit(xtr, ytr)
    acc = accuracy_from_voltage(xte @ w + b, yte)
    assert 0.45 <= acc <= 0.55
