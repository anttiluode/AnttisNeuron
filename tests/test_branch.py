import numpy as np
import pytest
from anttis_neuron.branch import LinearBranch, activation, decay_ratio


def test_unstable_branch_is_rejected():
    with pytest.raises(ValueError):
        LinearBranch(np.diag([1.01, 0.5]), np.ones(2), np.ones(2))


def test_square_activation():
    x = np.array([-2.0, 3.0])
    assert np.allclose(activation("square", x), np.array([4.0, 9.0]))


def test_slow_to_fast_mode_ratio_grows():
    r1 = decay_ratio(0.02, 0.2, 5)
    r2 = decay_ratio(0.02, 0.2, 20)
    assert r2 > r1 > 1.0


def test_linear_branch_step_matches_state_space_update():
    branch = LinearBranch(np.diag([0.8, 0.4]), np.array([1.0, 0.5]), np.array([1.0, -1.0]))
    out = branch.step(2.0)
    assert np.allclose(branch.state, np.array([2.0, 1.0]))
    assert np.isclose(out, 1.0)
