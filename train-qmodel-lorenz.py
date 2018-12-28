import numpy as np
import gym
from sys import path
path.insert(0, '..')
import lorenz
from lorenz.util import Transition
from lorenz.util import unfold
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
        self.stack_idx = 0

    def push(self, state, action, next_state, reward):
        """save a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.stack_idx] = Transition(state, action, next_state, reward)
        self.stack_idx = (self.stack_idx + 1) % self.capacity

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


def plot_episode_returns(returns, num_avg=100, stride=1):
    assert num_avg>stride
    plt.figure(2)
    plt.clf()
    returns = np.asarray(returns)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(returns)
    # Take episode averages and plot them too
    if len(returns) >= num_avg:
        means = unfold(returns, 0, num_avg, stride).mean(1)
        # means = np.concatenate((np.zeros(num_avg-stride), means))
        plt.plot((num_avg-stride)+np.arange(means.size), means)

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model(memory, policy_net, target_net, pnet_optimizer):
    """perform one optimization step on the policy network.

    1.  Sample the memory.  Each element has state $s_t$, action $a_t$, next
    state $s_{t+1}$, and reward $r_{t+1}$.

    2.  Compute the state-action fn, $Q(s_t, a_t)$, using `policy_net`.

    3.  Compute the value fn, $V(s_{t+1})=\max_{a} Q(s_{t+1}, a)$, using
    `target_net`.

    4.  Compute the expected state-action fn,
    $Q_e(s_t,a_t)=r_{t+1}+V(s_{t+1})\gamma$.  $\gamma$ is a discount.

    5.  Compute the Huber loss of $Q_e(s_t,a_t)-Q(s_t,a_t)$,
    and optimize using `pnet_optimizer`.
    """
    if len(memory) < BATCH_SIZE:
        return
    # Step 1:
    # get list of transitions and transpose it to
    # an `Transition` tuple of lists:
    list_of_transitions = memory.sample(BATCH_SIZE)
    batch = Transition.transpose(list_of_transitions)
    # Compute a mask of non-final states
    non_final_mask = torch.tensor(
            list(map(lambda s: s is not None, batch.next_state)),
            device=device, dtype=torch.uint8)
    # .. and concatenate the batch elements
    non_final_next_states = torch.cat([
        s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Step 2:
    # $Q(s_t, a_t)$
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Step 3:
    # $V(s_{s+1})=\max_{a} Q(s_{t+1}, a)$
    # (final states get value zero)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Step 4:
    # $Q_e(s_t,a_t) = r_{t+1} + \gamma V(s_{t+1})$
    expected_state_action_values = reward_batch + GAMMA * next_state_values
    # Step 5:
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the policy net
    pnet_optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    pnet_optimizer.step()


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

optimize_policy_net = optim.RMSprop(policy_net.parameters(), lr=1e-5)
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

        # Optimize the policy network (one step)
        optimize_model(memory, policy_net, target_net, optimize_policy_net)
        if done:
            env.render()
            episode_returns.append(episode_return)
            plot_episode_returns(episode_returns)
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
