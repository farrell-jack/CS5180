# Jack Farrell
# Reinforcement Learning
# Problem Set 0
# Problem 3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------
#     Part B
#-----------------------------

#------------------
# Random Policy
#------------------
class RandomPolicy:
    def __init__(self, T: int, seed=None):
        self.T = T
        self.rng = np.random.default_rng(seed)

    def act(self, obs) -> int:
        t, u = obs

        p_accept = 1.0 / self.T

        return 1 if self.rng.random() < p_accept else 0
    
#-----------------
# Threshold Policy
#-----------------
class ThresholdPolicy:
    def __init__(self, u_min: float):
        self.u_min = u_min

    def act(self, obs) -> int:
        t, u = obs
        return 1 if u >= self.u_min else 0
    
#----------------------------
# Optimal Policy Lookup Table
#----------------------------
class OptimalPolicy:
    def __init__(self, T: int, K: int):
        self.T = T
        self.K = K

        # initialize lookup table
        # rows: time t
        # cols: utility u in {1,...,K}
        self.policy_table = np.zeros((T + 1, K + 1), dtype=int)

        # w from question 1c
        w = {
        4: 3.4375,
        3: 3.25,
        2: 3.0,
        1: 2.5
        }

        # build policy from table
        for t in range(1, T + 1):
            w_next = w.get(t, 0)

            for u in range(1, K + 1):
                if u >= w_next:
                    self.policy_table[t, u] = 1  # accept
                else:
                    self.policy_table[t, u] = 0  # reject

    def act(self, obs) -> int:
        t, u = obs

        t = int(t)
        u = int(u)

        # safety clamp
        t = min(max(t, 0), self.T)
        u = min(max(u, 1), self.K)

        return int(self.policy_table[t, u])