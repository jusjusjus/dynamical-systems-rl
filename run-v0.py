import numpy as np
import gym
from sys import path
path.insert(0, '..')
import lorenz
from lorenz.util import Transition
import random
import math

import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

ENV_NAME = 'Lorenz-v0'
# Batch size is sampled from the experience
batch_size = 32

# Get the environment and extract the number of actions available in the Cartpole problem
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(env.observation_space.shape[0], 32)
        self.dense2 = nn.Linear(32, nb_actions)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        return self.dense2(x)


def plot_episode_returns():
    plt.figure(2)
    plt.clf()
    returns_tensor = torch.tensor(episode_returns, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(returns_tensor.numpy())
    # Take 100 episode averages and plot them too
    if len(returns_tensor) >= 100:
        means = returns_tensor.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states
    non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
    # .. and concatenate the batch elements
    non_final_next_states = torch.cat([
        s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-5)
memory = ReplayMemory(10000)

episode_returns = []

num_episodes = 200
for i_episode in range(num_episodes):
    # Initialize the environment and state
    episode_return = 0
    last_state = env.reset()
    last_state = torch.FloatTensor(last_state).unsqueeze(0)
    for t in count():
        # Select and perform an action
        action = select_action(last_state)
        next_state, reward, done, _ = env.step(action.item())
        episode_return += reward
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        reward = torch.FloatTensor([reward], device=device)

        # Enable this line to render every step
        # env.render()

        # Store the transition in memory
        memory.push(last_state, action, next_state, reward)

        # Move to the next state
        last_state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            env.render()
            episode_returns.append(episode_return)
            plot_episode_returns()
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
