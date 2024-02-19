import os
from datetime import datetime
import random

import numpy as np

from env import Actions, DeadlineEnv

total_test_episodes = 50000
max_ep_len = 30
num_actions = 3

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

    # scenario 2
    a11 = Actions((9, 1), (4, 1))
    a12 = Actions((3, 0.5), (2, 0.5))
    a13 = Actions((3, 0.5), (2, 0.5))

    a21 = Actions((3, 0.5), (2, 0.5))
    a22 = Actions((3, 0.5), (2, 0.5))
    a23 = Actions((10, 1), (5, 1))

    a31 = Actions((15, 1), (5, 1))
    a32 = Actions((10, 2), (5, 1))
    a33 = Actions((5, 1), (5, 1))
    env = DeadlineEnv(3, [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], 30)



    for cnt in range(1, 11):
        total_reward = 0
        for ep in range(1, total_test_episodes + 2):
            ep_reward = 0
            state = env.reset()

            action = 0
            tmp_cnt = cnt
            for t in range(1, max_ep_len + 2):
                if tmp_cnt == 0:
                    tmp_cnt = cnt
                    action += 1
                    action %= num_actions

                tmp_cnt -= 1
                state, reward, done = env.step(action)
                ep_reward += reward

                if done:
                    break

            total_reward += 0 if ep_reward < 0 else 1

        print(f"Count: {cnt}")
        print(f"Average Reward Per {total_test_episodes} Episodes: {float(total_reward)/total_test_episodes}")
        print()