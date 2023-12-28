import numpy as np
import random
from scipy.stats import norm


def normal(x, mean, dev):
    return norm.cdf(x, mean, dev)


class Actions:
    def __init__(self, P, E):
        # these will be maps mapping value -> probabilities
        self.P = P
        self.E = E


class DeadlineEnv:
    def __init__(self, n_plans, m_actions, deadline):
        self.n_plans = n_plans
        self.m_actions = m_actions
        self.deadline = deadline
        self.state = np.zeros(3 * n_plans + 1)
        self.action_space = np.arange(n_plans)

    def get_pt(self, ix):
        return ix + 1 + 2 * self.n_plans

    def get_et(self, ix):
        return ix + 1

    def get_ri(self, ix):
        return ix + 1 + self.n_plans

    def is_done(self):
        if self.state[0] > self.deadline:
            return True

        for i in range(self.n_plans):
            if self.state[self.get_ri(i)] == len(self.m_actions[i]):
                if self.state[0] + self.state[self.get_et(i)] <= self.deadline:
                    return True

        return False

    def get_state(self):
        return np.copy(self.state)

    def reset(self):
        self.state = np.zeros(3 * self.n_plans + 1)
        return self.get_state()

    def get_reward(self):
        if self.state[0] > self.deadline:
            return -1
        if not self.is_done():
            return 0
        return 1

    def update(self, action):
        self.state[0] += 1

        if self.state[self.get_ri(action)] == len(self.m_actions[action]):
            return self.get_state()

        # print(action, self.get_ri(action), self.m_actions)
        # p_dist = self.m_actions[action][round(self.state[self.get_ri(action)])].P
        # e_dist = self.m_actions[action][round(self.state[self.get_ri(action)])].E

        mu_p = self.m_actions[action][round(self.state[self.get_ri(action)])].P[0]
        sigma_p = self.m_actions[action][round(self.state[self.get_ri(action)])].P[1]
        mu = self.m_actions[action][round(self.state[self.get_ri(action)])].E[0]
        sigma = self.m_actions[action][round(self.state[self.get_ri(action)])].E[1]
        e_dist = round(np.random.normal(mu, sigma, 1)[0])

        rng = np.random.default_rng()

        # found = rng.random() <= p_dist(self.state[self.get_pt(action)] + 1)
        found = rng.random() <= normal(self.state[self.get_pt(action)] + 1, mu_p, sigma_p)

        if found:
            self.state[self.get_pt(action)] = 0
            self.state[self.get_ri(action)] += 1

            # x = np.random.choice(len(e_dist), p=e_dist)
            x = e_dist

            self.state[self.get_et(action)] += x
        else:
            self.state[self.get_pt(action)] += 1

        return self.get_state()

    def step(self, action):
        return self.update(action), self.get_reward(), self.is_done()
