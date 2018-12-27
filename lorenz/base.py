
import numpy as np

import gym
from gym import spaces
from .vis import Renderer
from gym.utils import seeding

class LorenzBase(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    
    limits = 50.0
    sigma = 10.0
    beta = 8/3
    rho = 28.0

    def __init__(self):
        self.dt = 0.1
        self.stride = 10
        self.dt_sim = self.dt/self.stride
        self.max_steps = 256
        self.viewer = None
        self.state = None
        self.time_step = 0
        self.seed()

    def derivs(self, state):
        x, y, z = state
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ], dtype=np.float32)

    @property
    def observable(self):
        return self.state

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self.viewer = self.viewer or Renderer(self.max_steps)
        self.viewer.update_data(self.history[:, :self.time_step])
        return self.viewer.render(mode)

    def close(self):
        if self.viewer: self.viewer.close()
