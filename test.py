from env import DeadlineEnv, Actions
from ppo import PPO
import numpy as np
import time
from datetime import datetime
import numpy as np


def linear(x):
    return x / 5.0

def expoential(x):
     return 1 - np.exp(-x / 2.0)

def twopow(x):
     return 1/256 * np.power(2, x)

def quadratic(x):
    return x**2 / 25.0

def logistic(x):
    mean = 0.3
    s = 3
    return 1 / (1 + np.exp(-(x - mean) / s))

def print_state(state, n_plans):
    print(state)
    print("Time Step: ", state[0])
    
    for i in range(n_plans):
        print(f"PLAN {i + 1}")
        print("Execution Time: ", state[i + 1])
        print("Last Found: ", state[i + n_plans + 1])
        print("Current Planning Time: ", state[i +  1 + 2 * n_plans])
        print("======================")


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving


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
    env = DeadlineEnv(4, [[a11, a12, a13], [a21, a22, a23], [a31, a32, a33], [a41, a42, a43]], 60)

    max_ep_len = 61         # max timesteps in one episode

    render = False              # render environment on screen
    frame_delay = 1            # if required; add delay b/w frames

    total_test_episodes = 1000    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    # state space dimension
    state_dim = env.state.shape[0]

    # action space dimension
    action_dim = len(env.m_actions)

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/deadline/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format('deadline', random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state = env.reset()

        for t in range(1, max_ep_len+1):
            # print_state(state, 2)
            action = ppo_agent.select_action(state)
            state, reward, done = env.step(action)
            ep_reward += reward

            if render:
                print("Model chose: ", action)
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0


    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    test()

