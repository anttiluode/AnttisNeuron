import numpy as np
from anttis_neuron.oja import absolute_cosine, oja_fit, principal_eigenvector


def test_principal_eigenvector_is_normalized():
    cov = np.diag([4.0, 2.0, 1.0])
    v = principal_eigenvector(cov)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert absolute_cosine(v, np.array([1.0, 0.0, 0.0])) > 0.999999


def test_oja_recovers_known_principal_direction():
    rng = np.random.default_rng(11)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    cov = q @ np.diag([8.0, 3.0, 1.0, 0.5]) @ q.T
    x = rng.multivariate_normal(np.zeros(4), cov, size=12000)
    learned = oja_fit(x, lr=0.002, epochs=3, seed=3)
    target = principal_eigenvector(cov)
    assert absolute_cosine(learned, target) >= 0.98
