from copy import deepcopy

from mcts_stoch import MCTS
from env import Actions, DeadlineEnv

def print_state(state):
    print(state, state.shape)
    print("Time Step: ", state[0])

    n_plans = (state.shape[0] - 1) // 3

    for i in range(n_plans):
        print(f"PLAN {i + 1}")
        print("Execution Time: ", state[i + 1])
        print("Last Found: ", state[i + n_plans + 1])
        print("Current Planning Time: ", state[i + 1 + 2 * n_plans])
        print("======================")


def play():
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
    state = DeadlineEnv(4, [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33], [a41, a42, a43]], 60)
    mcts = MCTS(state)

    while not state.is_done():
        print("Current state:")
        print(state.get_state())

        print("Thinking...")

        mcts.search(60)
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        move = mcts.best_move()

        print("MCTS chose move: ", move)

        state.step(move)
        mcts.move(move, deepcopy(state))

    if state.get_reward() > 0:
        print("Success")
    else:
        print("Failed")


if __name__ == '__main__':
    play()