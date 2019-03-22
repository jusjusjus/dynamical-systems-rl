
from os.path import dirname, join
import sys
sys.path.insert(0, join(dirname(__file__), '..'))

import numpy as np
from lorenz.v0 import LorenzEnv_v0

import pytest

@pytest.fixture
def env():
    return LorenzEnv_v0()

_action_strength = 50.0
@pytest.mark.parametrize("state, expected, action", [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 3),
    ([1.0, 0.0, 0.0], [-LorenzEnv_v0.sigma, LorenzEnv_v0.rho, 0.0], 3),
    ([0.0, 1.0, 0.0], [LorenzEnv_v0.sigma, -1.0, 0.0], 3),
    ([0.0, 0.0, 1.0], [0.0, 0.0, -LorenzEnv_v0.beta], 3),
    ([1.0, 1.0, 0.0], [_action_strength, LorenzEnv_v0.rho-1.0, 1.0], 0),
    ([0.0, 1.0, 1.0], [LorenzEnv_v0.sigma, _action_strength-1.0, -LorenzEnv_v0.beta], 1),
    ([1.0, 0.0, 1.0], [-LorenzEnv_v0.sigma, LorenzEnv_v0.rho-1.0, _action_strength-LorenzEnv_v0.beta], 2),
])
def test_ode(state, expected, action, env):
    """Test LorenzEnv_v0 with action strength of 50.0"""
    env.action_strength = _action_strength 
    state, expected = np.array(state, dtype=np.float32), np.array(expected, dtype=np.float32)
    deriv = env.derivs(state, action)
    assert all(deriv == expected)

def test_reset(env):
    steps = 4
    env.reset()
    for action in range(steps):
        env.step(action)
    assert env.time_step == steps
    assert not all(env.history.flatten() == 0.0)
    last_state = env.state
    env.reset()
    assert all(env.history.flatten() == 0.0)
    assert not all(last_state == env.state)


