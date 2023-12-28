from env import Actions, DeadlineEnv
import numpy as np


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
    a11 = Actions((15, 2), (15, 2))
    a12 = Actions((15, 2), (15, 2))
    a13 = Actions((20, 3), (20, 3))

    a21 = Actions((30, 3), (10, 2))
    a22 = Actions((15, 2), (20, 2))
    a23 = Actions((30, 3), (20, 2))

    a31 = Actions((20, 2), (60, 5))
    a32 = Actions((10, 2), (5, 1))
    a33 = Actions((10, 2), (5, 1))

    a41 = Actions((10, 2), (5, 1))
    a42 = Actions((10, 2), (5, 1))
    a43 = Actions((40, 4), (25, 3))
    env = DeadlineEnv(4, [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33], [a41, a42, a43]], 100)

    won = False
    cur_state = env.reset()
    while True:
        print_state(cur_state, 4)
        print()

        action = int(input("choose your action: "))

        new_obs, reward, done = env.step(action)
        print("Reward: ", reward)
        cur_state = new_obs
        if done:
            print_state(cur_state, 4)
            if reward == 1:
                won = True
            break

    if won:
        print("You made it")
    else:
        print("You did not make it")
