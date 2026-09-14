"""AnttisNeuron: a minimal spectral two-layer neuron research model."""

from .branch import LinearBranch, activation, decay_ratio, spectral_radius
from .model import (
    accuracy_from_voltage,
    collapsed_linear_filter,
    quadratic_matrix,
    ridge_fit,
    static_features,
    static_voltage,
)
from .oja import absolute_cosine, oja_fit, principal_eigenvector

__all__ = [
    "LinearBranch",
    "absolute_cosine",
    "accuracy_from_voltage",
    "activation",
    "collapsed_linear_filter",
    "decay_ratio",
    "oja_fit",
    "principal_eigenvector",
    "quadratic_matrix",
    "ridge_fit",
    "spectral_radius",
    "static_features",
    "static_voltage",
]
