import os
from datetime import datetime
import random

import numpy as np

from env import Actions, DeadlineEnv

total_test_episodes = 10000
max_ep_len = 21
num_actions = 2


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


def compute_success(plan_i, action_i, timestep, execution_time, actions, shared, deadline, p_norm=1):
    if action_i == len(actions[plan_i]) - 1:
        # print("Inside")
        proba = 0.0
        for t_inner in range(1, deadline - timestep + 1):
            density_planning = cumulative_to_density(actions[plan_i][action_i].P)
            # print(density_planning, actions[plan_i][action_i].E)
            if t_inner not in density_planning:
                continue

            proba += density_planning[t_inner] * get_proba_under(actions[plan_i][action_i].E, deadline - t_inner - timestep - execution_time)

        if f'{plan_i}_{action_i}_{timestep}_{execution_time}' not in dp:
            dp[f'{plan_i}_{action_i}_{timestep}_{execution_time}'] = {}

        dp[f'{plan_i}_{action_i}_{timestep}_{execution_time}'][tuple(shared)] = proba

        # print(proba)
        return proba

    if f'{plan_i}_{action_i}_{timestep}_{execution_time}' in dp and tuple(shared) in dp[f'{plan_i}_{action_i}_{timestep}_{execution_time}']:
        return dp[f'{plan_i}_{action_i}_{timestep}_{execution_time}'][tuple(shared)]

    proba = 0.0
    density_planning = cumulative_to_density(actions[plan_i][action_i].P)
    # print(density_planning)
    for t_inner in range(1, deadline - timestep + 1):
        if t_inner not in density_planning:
            continue

        for e_t in actions[plan_i][action_i].E:
            for s_i in shared:
                proba += pow(compute_success(s_i, action_i+1, timestep + t_inner, execution_time + e_t, actions, actions[s_i][action_i+1].shared, deadline, p_norm), p_norm)

    proba = pow(proba, 1.0/p_norm)

    if f'{plan_i}_{action_i}_{timestep}_{execution_time}' not in dp:
        dp[f'{plan_i}_{action_i}_{timestep}_{execution_time}'] = {}

    dp[f'{plan_i}_{action_i}_{timestep}_{execution_time}'][tuple(shared)] = proba

    return proba


# dp[i][j][k] = probability of success at action i, current timestep j, and cumulative execution time so far
if __name__ == '__main__':
    # scenario 4
    a11 = Actions({8: 1}, {5: 1}, [0, 1, 2])
    a12 = Actions({1: 0.8, 20: 1}, {1: 0.4, 8: 0.7}, [0])

    a21 = Actions({8: 1}, {5: 1}, [0, 1, 2])
    a22 = Actions({1: 1}, {1: 0.3, 8: 0.7}, [1])

    a31 = Actions({8: 1}, {5: 1}, [0, 1, 2])
    a32 = Actions({1: 0.8, 20: 1}, {1: 0.4, 8: 0.7}, [2])

    a41 = Actions({8: 1}, {5: 1}, [3])
    a42 = Actions({1: 1}, {1: 0.35, 8: 0.65}, [3])
    env = DeadlineEnv(4, [[a11, a12], [a21, a22], [a31, a32], [a41, a42]], 20)

    proba_success = {}

    for i in range(env.n_plans):
        proba_success[i] = compute_success(i, 0, 0, 0, env.m_actions, env.m_actions[i][0].shared, 20, 500)


    print(proba_success)

    total_reward = 0
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len + 1):
            action = max(proba_success, key=lambda k: proba_success[k])
            state, reward, done = env.step(action)
            ep_reward += reward

            if done:
                break

        total_reward += 0 if ep_reward < 0 else 1

    print(f"Average Reward Per {total_test_episodes} Episodes: {float(total_reward) / total_test_episodes}")
