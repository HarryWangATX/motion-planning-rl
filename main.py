from env import Actions, DeadlineEnv
import numpy as np
from utils import get_environments


def linear(x):
    return x / 5.0


def quadratic(x):
    return x ** 2 / 25.0


def logistic(x):
    mean = 0.3
    s = 3
    return 1 / (1 + np.exp(-(x - mean) / s))


def print_state(state, n_plans):
    print(state, state.shape)
    print("Time Step: ", state[0])

    for i in range(n_plans):
        print(f"PLAN {i + 1}")
        print("Execution Time: ", state[i + 1])
        print("Last Found: ", state[i + n_plans + 1])
        print("Current Planning Time: ", state[i + 1 + 2 * n_plans])
        print("======================")


if __name__ == "__main__":
    a11 = Actions({1: 0.1, 2: 0.3, 5: 0.7, 8: 0.9, 9: 1}, {1: 1}, [0])
    env = DeadlineEnv(1, [[a11]], 9)

    env.step(0)
    print(env.get_reward())
    env.new_times()
    print(env.planning_times)


    envs = get_environments()

    env_track = envs[0]

    tally = {}

    for i in range(10000):
        env_track.reset()
        if env_track.planning_times[1][0] not in tally:
            tally[env_track.planning_times[1][0]] = 0
        tally[env_track.planning_times[1][0]]+=1

    print(tally)

    # won = False
    # cur_state = env.reset()
    # while True:
    #     print_state(cur_state, 2)
    #     print('\n\n\n')
    #
    #     action = int(input("choose your action: "))
    #
    #     new_obs, reward, done = env.step(action, debug=True)
    #     print("Reward: ", reward)
    #     cur_state = new_obs
    #     if done:
    #         print_state(cur_state, 2)
    #         if reward == 1:
    #             won = True
    #         break
    #
    # if won:
    #     print("You made it")
    # else:
    #     print("You did not make it")
