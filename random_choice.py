import os
from datetime import datetime
import random

import torch
import numpy as np

from env import Actions, DeadlineEnv

total_test_episodes = 100000
max_ep_len = 60

if __name__ == '__main__':
    a11 = Actions((5, 1), (5, 1))
    a12 = Actions((5, 1), (5, 1))
    a13 = Actions((7, 1), (7, 1))

    a21 = Actions((10, 1), (30, 1))
    a22 = Actions((20, 1), (7, 1))
    a23 = Actions((10, 2), (7, 1))

    a31 = Actions((7, 1), (20, 2))
    a32 = Actions((3, 1), (30, 1))
    a33 = Actions((3, 1), (20, 1))

    a41 = Actions((3, 1), (3, 1))
    a42 = Actions((3, 1), (3, 1))
    a43 = Actions((13, 2), (80, 1))
    env = DeadlineEnv(4, [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33], [a41, a42, a43]], max_ep_len)


    total_reward = 0
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            action = random.randint(0, 3)
            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                break

        total_reward += ep_reward

    print(f"Average Reward Per {total_test_episodes} Episodes: {float(total_reward)/total_test_episodes}")