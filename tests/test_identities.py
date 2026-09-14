import numpy as np
from anttis_neuron.model import (
    collapsed_linear_filter,
    quadratic_matrix,
    static_features,
    static_voltage,
)


def test_linear_branches_collapse_to_one_filter():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 5))
    m = rng.normal(size=(4, 5))
    a = rng.normal(size=4)
    bias = 0.3
    direct = static_voltage(x, m, a, bias=bias, nonlinearity="identity")
    collapsed = x @ collapsed_linear_filter(m, a) + bias
    assert np.allclose(direct, collapsed, atol=1e-12)


def test_static_model_equals_explicit_shallow_network():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(24, 6))
    m = rng.normal(size=(5, 6))
    a = rng.normal(size=5)
    bias = -0.2
    direct = static_voltage(x, m, a, bias=bias, nonlinearity="nmda")
    hidden = static_features(x, m, nonlinearity="nmda")
    explicit = hidden @ a + bias
    assert np.allclose(direct, explicit, atol=1e-12)


def test_square_branches_equal_quadratic_form():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(16, 6))
    m = rng.normal(size=(3, 6))
    a = rng.normal(size=3)
    direct = static_voltage(x, m, a, bias=0.0, nonlinearity="square")
    q = quadratic_matrix(m, a)
    quadratic = np.einsum("bi,ij,bj->b", x, q, x)
    assert np.allclose(direct, quadratic, atol=1e-12)
