import random
import time
from copy import deepcopy

import numpy as np

from env import Actions, DeadlineEnv

from collections import defaultdict
import math


class MCTSNode:
    def __init__(self, state_or_move=None, parent=None, action: bool = False):
        self.state_or_move = state_or_move
        self.children = {}
        self.parent = parent
        self.N = 0.0
        self.Q = 0.0
        self.action = action
        self.outcome = 0  # 1 if successful, -1 if failed, 0 if intermediate

    def value(self, exploration_param=0.5):
        if self.N == 0:
            return 0 if exploration_param == 0 else float('inf')
        return self.Q / self.N + exploration_param * math.sqrt(math.log2(self.parent.N) / self.N)


class MCTS:
    def __init__(self, state: DeadlineEnv, num_rollouts=1000):
        self.root_state = deepcopy(state)
        self.root = MCTSNode()
        self.num_rollouts = num_rollouts
        self.run_time = 0
        self.node_count = 0

    def select_node(self) -> tuple[MCTSNode, DeadlineEnv]:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            # descend to the maximum value node, break ties at random
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value
            max_nodes = [n for n in node.children.values() if n.value == max_value]
            node = random.choice(max_nodes)

            state.step(node.state_or_move)

            if node.N == 0:
                return node, state

            state_tup = tuple(state.get_state())

            if state_tup not in node.children:
                state_child = MCTSNode(state_or_move=None, parent=node, action=False)
                node.children[state_tup] = state_child
            else:
                state_child = node.children[state_tup]

            node = state_child

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))

        return node, state

    @staticmethod
    def expand(parent: MCTSNode, state: DeadlineEnv) -> bool:
        if state.is_done():
            return False

        for i in range(state.n_plans):
            parent.children[i] = MCTSNode(state_or_move=i, parent=parent, action=True)

        return True

    @staticmethod
    def roll_out(state: DeadlineEnv) -> int:

        moves = np.arange(state.n_plans)  # Get a list of all possible moves in current state of the env
        state.new_times()
        reward = state.get_reward()
        while not state.is_done():
            move = random.choice(moves)
            _, reward, _ = state.step(move)

        return reward

    @staticmethod
    def backup(node: MCTSNode, reward: int) -> None:
        # Careful: The reward is calculated for player who just played
        # at the node and not the next player to play
        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent

    def search(self, time_budget: int) -> None:
        start_time = time.process_time()
        num_rollouts = 0

        # do until we exceed our time budget
        while time.process_time() - start_time < time_budget:
            node, state = self.select_node()

            outcome = self.roll_out(state)
            self.backup(node, outcome)
            num_rollouts += 1
        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self) -> int:
        if self.root_state.is_done():
            return -1

        # choose the move of the most simulated node breaking ties randomly
        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)
        return best_child.state_or_move

    def move(self, action: int, new_state: DeadlineEnv) -> None:
        self.root = self.root.children[action]
        state_tup = tuple(new_state.get_state())

        # print(action, state_tup)

        if state_tup in self.root.children:
            self.root = self.root.children[state_tup]
        else:
            self.root = MCTSNode(state_or_move=None, parent=None, action=False)

        self.root_state = deepcopy(new_state)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time
