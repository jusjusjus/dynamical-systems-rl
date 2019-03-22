
from os.path import dirname, join
import sys
sys.path.insert(0, join(dirname(__file__), '..'))

import numpy as np
from lorenz.util import Transition, unfold

import pytest

def test_Transition_transpose():
    """test method Transition.transpose in forward direction"""
    vals = list(range(5))
    transitions = [
        Transition(state=v, action=v, next_state=v, reward=v)
        for v in vals
    ]
    transposed = Transition.transpose(transitions)
    for v in transposed:
        assert all(vi == vj for vi, vj in zip(v, vals))

def test_Transition_transpose2():
    """test method Transition.transpose in backward direction"""
    vals = list(reversed(range(5)))
    transitions = Transition(
            state=vals, action=vals, next_state=vals, reward=vals)
    transposed = Transition.transpose(transitions)
    for i, trans in zip(vals, transposed):
        assert all(i == t for t in trans)

def test_Transition_transpose3():
    """test method Transition.transpose in backward direction"""
    vals = list(reversed(range(5)))
    transitions = Transition(
            state=vals, action=vals, next_state=vals, reward=vals)
    transposed = Transition.transpose(Transition.transpose(transitions))
    for ition, posed in zip(transitions, transposed):
        assert tuple(ition) == posed


@pytest.mark.parametrize("arr, dim, size, step, expected, dtype", [
    ([0, 1, 2, 3, 4, 5], 0, 2, 1, [0, 1, 1, 2, 2, 3, 3, 4, 4, 5], np.float32),
    ([0, 1, 2, 3, 4, 5], 0, 3, 1, [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5], np.float32),
    ([0, 1, 2, 3, 4, 5], 0, 2, 1, [0, 1, 1, 2, 2, 3, 3, 4, 4, 5], np.float64),
    ([0, 1, 2, 3, 4, 5], 0, 3, 1, [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5], np.float64),
])
def test_unfold(arr, dim, size, step, expected, dtype):
    arr = np.asarray(arr, dtype=dtype)
    expected = np.asarray(expected, dtype=dtype)
    unfolded = unfold(arr, dim, size, step).flatten()
    assert all(unfolded == expected)
