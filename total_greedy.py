import os
from datetime import datetime
import random

import numpy as np

from utils import get_environments

total_test_episodes = 10000
max_ep_len = 10
num_actions = 2


def weighted_average(cumulative_probabilities_map):
    cumulative_probabilities = sorted(cumulative_probabilities_map.values())
    values = list(cumulative_probabilities_map.keys())
    # print(cumulative_probabilities)
    # print(values)

    weighted_sum = 0
    total_weight = cumulative_probabilities[-1]

    for i in range(len(cumulative_probabilities)):
        weight = cumulative_probabilities[i]
        if i > 0:
            weight -= cumulative_probabilities[i - 1]

        weighted_sum += weight * values[i]

    return weighted_sum / total_weight


def weighted_average_density(density_map):
    values = list(density_map.keys())
    probabilities = list(density_map.values())

    total_weighted_sum = sum(value * probability for value, probability in zip(values, probabilities))
    total_weight = sum(probabilities)

    return total_weighted_sum / total_weight


def compute_total_mean(env):
    plan_times = {}
    for pi in range(len(env.m_actions)):
        total_time = 0
        for ai in range(len(env.m_actions[pi])):
            ai_p = env.m_actions[pi][ai]
            total_time += weighted_average(ai_p.P)
            total_time += weighted_average_density(ai_p.E)
        plan_times[pi] = total_time
    return plan_times


if __name__ == '__main__':
    envs = get_environments()

    for index, env in enumerate(envs):
        plan_times = compute_total_mean(env)

        print(plan_times)
        total_reward = 0
        for ep in range(1, total_test_episodes + 1):
            ep_reward = 0
            state = env.reset()

            for t in range(1, env.deadline + 1):
                action = min(plan_times, key=lambda k: plan_times[k])
                state, reward, done = env.step(action)
                ep_reward += reward

                if done:
                    break

            total_reward += 0 if ep_reward < 0 else 1

        print(f"Environment {index + 1} - Average Reward Per {total_test_episodes} Episodes: {float(total_reward) / total_test_episodes}")
