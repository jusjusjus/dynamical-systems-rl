
import numpy as np

import gym
from . import v0
from .v0 import LorenzEnv_v0

gym.envs.registry.register(
    id='Lorenz-v0',
    entry_point='lorenz:LorenzEnv_v0',
    max_episode_steps=1024,
    reward_threshold=10.0
)
