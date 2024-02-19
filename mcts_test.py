from copy import deepcopy

from mcts_stoch import MCTS
from env import Actions, DeadlineEnv
from total_greedy import compute_total_mean
from utils import get_environments

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
    mcts_print = True
    envs = get_environments()

    total_episode = 100

    for index, tmp_env in enumerate(envs):
        total_success = 0
        mean_times = compute_total_mean(tmp_env)
        allocated_time = min(mean_times.values(), key=lambda x: x)
        for _ in range(total_episode):
            env = deepcopy(tmp_env)
            env.reset()
            print(env.planning_times)
            mcts = MCTS(deepcopy(env))

            while not env.is_done():
                if mcts_print:
                    print("Current state:")
                    print(env.get_state())

                    print("Thinking...")

                mcts.search(round(allocated_time))
                if mcts_print:
                    num_rollouts, run_time = mcts.statistics()
                    print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
                move = mcts.best_move()

                if mcts_print:
                    print("MCTS chose move: ", move)

                env.step(move, debug=mcts_print)
                mcts.move(move, deepcopy(env))

                if mcts_print:
                    print('\n\n')

            if env.get_reward() > 0:
                total_success += 1

            if mcts_print and env.get_reward() > 0:
                print("Success")
            elif mcts_print:
                print("Failed")

        print(f"Environment {index + 1} - Average Reward Per {total_episode} Episodes: {float(total_success) / total_episode}")




if __name__ == '__main__':
    play()