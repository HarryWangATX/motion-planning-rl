import os
from datetime import datetime
import random
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import numpy as np

from env import Actions, DeadlineEnv

total_test_episodes = 10000
max_ep_len = 10
num_actions = 1

if __name__ == '__main__':
    # scenario 5
    a11 = Actions({1: 0.3, 5: 0.7, 9: 1.0}, {1: 0.25, 3: 0.5, 5: 0.25}, [0])
    env = DeadlineEnv(1, [[a11]], 6)

    total_reward = 0
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            action = 0 # min(plan_times, key=lambda k: plan_times[k])
            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                break

        # print(ep_reward)
        total_reward += 0 if ep_reward < 0 else 1

    print(f"Average Reward Per {total_test_episodes} Episodes: {float(total_reward) / total_test_episodes}")

    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    axs[0].hist(env.p_sampled, bins=5)
    axs[1].hist(env.e_sampled, bins=3)
    plt.savefig('Distribution')
