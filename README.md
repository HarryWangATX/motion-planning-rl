To run a progam, use `python {file}`, for example, `python dp_rechoose.py`, which will run the DP Rechoose strategy on all of the environments.

# Environments

```python
from env import Actions, DeadlineEnv

# Example: two planners, one fast but unreliable, one slow but sure
a_fast = Actions({1: 0.2, 3: 1.0}, {2: 0.5, 5: 1.0}, shared=[0])
a_slow = Actions({2: 1.0}, {3: 1.0}, shared=[1])
env = DeadlineEnv(
    n_plans=2,
    m_actions=[[a_fast], [a_slow]],
    deadline=10
)
```


## 🧠 Strategies & Implementations

This repository implements a variety of planning strategies—ranging from simple baselines to advanced RL and search methods. Each lives in its own module; here’s what they do and how they work.

---

1. Random Choice (random_choice.py)
What it does: At every decision step, this strategy picks one of the available plans uniformly at random.

2. Total Greedy (total_greedy.py)
What it does: This strategy always chooses the plan with the highest expected probability of completion within the remaining time.

3. Round Robin (round_robin.py)
What it does: This strategy cycles through available plans in a fixed order, taking one action from each plan per turn.

4. Greedy Dynamic Programming (greedy_dp.py & greedy_dp_better.py)
What they do: These modules use a one-shot dynamic programming (DP) approach to compute the best plan choice at each state defined by time and remaining actions, assuming no re-selection is possible.

5. DP with Re-selection (dp_rechoose.py & dp_rechoose2.py)
What they do: These methods extend the DP approach to allow re-selecting a plan after each action, recomputing the optimal next step at each decision point.

6. Monte Carlo Tree Search
Stochastic MCTS (mcts_stoch.py)
What it does: This strategy builds a search tree of simulated rollouts, using the Upper Confidence Bound for Trees (UCT) algorithm at each node to balance exploration and exploitation.

Deterministic Play (mcts_play.py)
What it does: After a search is performed, this method selects the child with the highest visit count at each real step, removing the stochasticity from the action selection.

7. Proximal Policy Optimization (PPO) (ppo.py & train.py)
What it does: This strategy learns a parameterized stochastic policy 
pi_x(a∣s) using a clipped surrogate objective.
