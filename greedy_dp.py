import os
from datetime import datetime
import random

import numpy as np

from utils import get_environments

total_test_episodes = 10000
max_ep_len = 14
num_actions = 4


def weighted_average(cumulative_probabilities_map):
    cumulative_probabilities = sorted(cumulative_probabilities_map.values())
    values = list(cumulative_probabilities_map.keys())

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


def cumulative_to_density(cumulative_probabilities_map):
    density_probabilities_map = {}

    cumulative_probs = sorted(cumulative_probabilities_map.values())
    values = list(cumulative_probabilities_map.keys())

    for i in range(len(cumulative_probs)):
        density_prob = cumulative_probs[i]
        if i > 0:
            density_prob -= cumulative_probs[i - 1]

        density_probabilities_map[values[i]] = density_prob

    return density_probabilities_map

def get_proba_under(execution, t):
    proba = 0.0
    for e_t in execution:
        if e_t <= t:
            proba += execution[e_t]

    return proba


dp = {}


def compute_success(plan_i, timestep, execution_time, actions, deadline):
    if plan_i == len(actions) - 1:
        # print("Inside")
        proba = 0.0
        for t_inner in range(1, deadline - timestep + 1):
            density_planning = cumulative_to_density(actions[plan_i].P)
            # print(density_planning, actions[plan_i].E)
            if t_inner not in density_planning:
                continue

            proba += density_planning[t_inner] * get_proba_under(actions[plan_i].E, deadline - t_inner - timestep - execution_time)

        dp[f'{plan_i}_{timestep}_{execution_time}'] = proba
        # print(proba)
        return proba

    if f'{plan_i}_{timestep}_{execution_time}' in dp:
        return dp[f'{plan_i}_{timestep}_{execution_time}']

    proba = 0.0
    density_planning = cumulative_to_density(actions[plan_i].P)
    # print(density_planning)
    for t_inner in range(1, deadline - timestep + 1):
        if t_inner not in density_planning:
            continue

        for e_t in actions[plan_i].E:
            proba += density_planning[t_inner] * actions[plan_i].E[e_t] * \
                     compute_success(plan_i + 1, timestep + t_inner, execution_time + e_t, actions, deadline)

    dp[f'{plan_i}_{timestep}_{execution_time}'] = proba
    return proba


# dp[i][j][k] = probability of success at action i, current timestep j, and cumulative execution time so far
if __name__ == '__main__':

    envs = get_environments()

    for index, env in enumerate(envs):
        proba_success = {}

        for i in range(env.n_plans):
            dp = {}
            proba_success[i] = compute_success(0, 0, 0, actions=env.m_actions[i], deadline=env.deadline)

        print(proba_success)

        total_reward = 0
        for ep in range(1, total_test_episodes + 1):
            ep_reward = 0
            state = env.reset()

            for t in range(1, env.deadline + 1):
                action = max(proba_success, key=lambda k: proba_success[k])
                state, reward, done = env.step(action)
                ep_reward += reward

                if done:
                    break

            total_reward += 0 if ep_reward <= 0 else 1

        print(f"Environment {index + 1} - Average Reward Per {total_test_episodes} Episodes: {float(total_reward) / total_test_episodes}")
