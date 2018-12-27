
import numpy as np
from gym import spaces
from .base import LorenzBase

_NO_ACTION = 3

class LorenzEnv_v0(LorenzBase):

    action_strength = 30.0
    actions = np.vstack([
        np.eye(3, dtype=np.float32),
        np.array([0, 0, 0], dtype=np.float32)
    ])

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-self.limits, self.limits, shape=(3,))
        self.history = np.zeros((6, self.max_steps), dtype=np.float32)

    def derivs(self, state, action=_NO_ACTION):
        return super().derivs(state) + \
                self.action_strength * self.actions[action]

    def reward(self, action):
        """return reward of observing a state and performing an action
        
        The reward is `0.5*dt` for positive `x` values, scaled with integration
        time step `dt`.  The reward is more positive if the idle action
        `_NO_ACTION` is taken during evalutation.  This reinforces low-energy
        solutions.
        """
        return (self.dt*(0.5+0.5*(action==_NO_ACTION))) if self.state[0]>0.0 else -self.dt

    def step(self, action=_NO_ACTION):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        for _ in range(self.stride):
            self.state += self.dt_sim * self.derivs(self.state, action)
        self.history[:, self.time_step] = np.concatenate(
                [self.state, self.actions[action]])
        self.time_step += 1
        done = self.time_step == self.max_steps
        return self.observable, self.reward(action), done, {}

    def reset(self):
        self.history = np.zeros((6, self.max_steps), dtype=np.float32)
        self.time_step = 0
        self.state = np.array(self.np_random.uniform(low=-1.0, high=1.0, size=(3,)), dtype=np.float32)
        return self.observable
