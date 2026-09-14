"""Gate 1: exact linear-collapse, shallow-network, and quadratic identities."""
from __future__ import annotations

import numpy as np

from anttis_neuron.branch import activation
from anttis_neuron.model import collapsed_linear_filter, quadratic_matrix, static_voltage
from experiments.common import main


def run(seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(256, 9))
    filters = rng.normal(size=(7, 9))
    out = rng.normal(size=7)
    bias = float(rng.normal())

    linear = static_voltage(x, filters, out, bias=bias, nonlinearity="identity")
    collapsed = x @ collapsed_linear_filter(filters, out) + bias

    hidden = activation("nmda", x @ filters.T)
    shallow = hidden @ out + bias
    direct = static_voltage(x, filters, out, bias=bias, nonlinearity="nmda")

    square = static_voltage(x, filters, out, bias=0.0, nonlinearity="square")
    q = quadratic_matrix(filters, out)
    quadratic = np.einsum("bi,ij,bj->b", x, q, x)

    return {
        "gate": 1,
        "seed": seed,
        "linear_collapse_max_abs_error": float(np.max(np.abs(linear - collapsed))),
        "two_layer_max_abs_error": float(np.max(np.abs(direct - shallow))),
        "quadratic_max_abs_error": float(np.max(np.abs(square - quadratic))),
    }


if __name__ == "__main__":
    main(run)
