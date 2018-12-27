
import numpy as np
import gym
import lorenz
from lorenz.util import unfold
from itertools import count
import matplotlib.pyplot as plt
ENV_NAME = 'Lorenz-v0'
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)


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

episode_returns = []
num_episodes = 100
for episode_idx in range(num_episodes):
    episode_return = 0
    last_state = env.reset()
    for t in count():
        # action = np.random.randint(4)
        action = lorenz.v0._NO_ACTION
        next_state, reward, done, _ = env.step(action)
        episode_return += reward
        env.render()
        if done:
            episode_returns.append(episode_return)
            plot_episode_returns(episode_returns, num_avg=10)
            break

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
