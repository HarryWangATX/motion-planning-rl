import numpy as np
import random
from scipy.stats import norm


def normal(x, mean, dev):
    return norm.cdf(x, mean, dev)


class Actions:
    def __init__(self, P, E, shared):
        # these will be maps mapping value -> probabilities
        self.P = P
        self.E = E
        self.shared = shared


class DeadlineEnv:
    def __init__(self, n_plans, m_actions, deadline):
        self.n_plans = n_plans
        self.m_actions = m_actions
        self.deadline = deadline
        self.state = np.zeros(3 * n_plans + 1)
        self.action_space = np.arange(n_plans)
        self.planning_times = []

        for i in range(0, n_plans):
            self.planning_times.append([])

        for i in range(0, n_plans):
            for j in range(0, len(m_actions[i])):
                self.planning_times[i].append(self.map_sample_cum(m_actions[i][j].P))

        self.e_sampled = []
        self.p_sampled = []

    def get_pt(self, ix):
        return ix + 1 + 2 * self.n_plans

    def get_et(self, ix):
        return ix + 1

    def get_ri(self, ix):
        return ix + 1 + self.n_plans

    def is_done(self):
        if self.state[0] >= self.deadline:
            return True

        for i in range(self.n_plans):
            if self.state[self.get_ri(i)] == len(self.m_actions[i]):
                if self.state[0] + self.state[self.get_et(i)] <= self.deadline:
                    return True

        return False

    def get_state(self):
        return np.copy(self.state)

    def make_shared_equal(self):
        for i in range(self.n_plans):
            for j in range(len(self.m_actions[i])):
                for s_i in self.m_actions[i][j].shared:
                    self.planning_times[s_i][j] = self.planning_times[i][j]

    def new_times(self):
        # given current state, conditional probabilities for outcomes
        self.planning_times = []

        for i in range(0, self.n_plans):
            self.planning_times.append([])

        # print("N_plans: ", self.n_plans)
        # print("M_actions: ", len(self.m_actions[0]))

        for i in range(0, self.n_plans):
            for j in range(0, len(self.m_actions[i])):
                # print(i, j)
                self.planning_times[i].append(self.map_get_given_state(self.m_actions[i][j].P, i, j))

        self.make_shared_equal()

    def reset(self):
        self.state = np.zeros(3 * self.n_plans + 1)
        self.planning_times = []

        for i in range(0, self.n_plans):
            self.planning_times.append([])

        for i in range(0, self.n_plans):
            for j in range(0, len(self.m_actions[i])):
                self.planning_times[i].append(self.map_sample_cum(self.m_actions[i][j].P))

        self.make_shared_equal()

        return self.get_state()

    def get_reward(self):
        if not self.is_done():
            return 0

        for i in range(self.n_plans):
            if self.state[self.get_ri(i)] == len(self.m_actions[i]):
                if self.state[0] + self.state[self.get_et(i)] <= self.deadline:
                    return 1

        return -1

    def map_sample(self, execution_map):
        random_num = random.random()

        # print("Random Execution: ", random_num)

        cur_sum = 0
        for e_i in execution_map:
            cur_sum += execution_map[e_i]
            if random_num <= cur_sum:
                return e_i
        return -1

    def map_sample_cum(self, planning_map):
        random_num = random.random()

        for p_i in planning_map:
            if planning_map[p_i] >= random_num:
                return p_i

        return -1

    def map_get_given_state(self, planning_map, plan, action):
        sumsum = 0.0
        cur_act = round(self.get_state()[self.get_ri(plan)])
        cur_plan = round(self.get_state()[self.get_pt(plan)])

        # print(planning_map)
        # print(action, cur_act, cur_plan)

        if action < cur_act:
            return -1
        elif action > cur_act:
            return self.map_sample_cum(planning_map)

        for p_i in planning_map:
            if p_i <= cur_plan:
                sumsum = planning_map[p_i]

        has = 1.0 - sumsum

        # print(has)

        new_planning = {}

        prev = 0.0
        total = 0.0
        mx = 0
        for p_i in planning_map:
            mx = max(mx, p_i)
            if p_i <= cur_plan:
                prev = planning_map[p_i]
                continue

            # print(planning_map[p_i] - prev, (planning_map[p_i] - prev) / has)
            prob = (planning_map[p_i] - prev) / has
            total += prob

            new_planning[p_i] = total
            prev = planning_map[p_i]

        if mx not in new_planning:
            new_planning[100000000] = 1
        elif new_planning[mx] != 1:
            new_planning[mx] = 1

        # print(new_planning)

        return self.map_sample_cum(new_planning)

    def update(self, action, debug=False):
        self.state[0] += 1
        cur_ri = int(self.state[self.get_ri(action)])

        if cur_ri == len(self.m_actions[action]):
            return self.get_state()

        e_dist = self.map_sample(self.m_actions[action][cur_ri].E)


        # found = rng.random() <= p_dist(self.state[self.get_pt(action)] + 1)
        next_pt = self.state[self.get_pt(action)] + 1
        # pmf = self.m_actions[action][cur_ri].P[next_pt] if next_pt in self.m_actions[action][cur_ri].P else -1
        # random_num = random.random()
        # print("Random: ", random_num)
        # found = random_num <= pmf
        found = False
        if next_pt == self.planning_times[action][cur_ri]:
            found = True

        if found:
            self.p_sampled.append(next_pt)
        self.e_sampled.append(e_dist)

        if debug:
            print("Next Timestep: ", next_pt)
            # print("PMF: ", pmf)
            print("Found: ", found)
            # print("Planning Map: ", self.m_actions[action][cur_ri].P)
            print("Execution Map: ", self.m_actions[action][cur_ri].E)
            print("Sampled Execution Time: ", e_dist)
            print("Shared: ", self.m_actions[action][cur_ri].shared)

        if found:
            x = e_dist
            for p in self.m_actions[action][cur_ri].shared:
                self.state[self.get_pt(p)] = 0
                self.state[self.get_ri(p)] += 1
                self.state[self.get_et(p)] += x
        else:
            for p in self.m_actions[action][cur_ri].shared:
                self.state[self.get_pt(p)] += 1

        return self.get_state()

    def step(self, action, debug=False):
        return self.update(action, debug=debug), self.get_reward(), self.is_done()
