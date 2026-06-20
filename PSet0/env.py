# Jack Farrell
# Reinforcement Learning
# Problem Set 0
# Problem 3

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

#-----------------------------
#     Part A
#-----------------------------
class ApartmentEnv(gym.Env):

    #--------------
    # init function
    #--------------
    def __init__(self, T: int, K: int, seed=None):
        super().__init__()

        self.T = T
        self.K = K

        # Random generator
        self.rng = np.random.default_rng(seed)

        # -------------------------
        # ACTION SPACE (accept or reject)
        # -------------------------
        self.action_space = spaces.Discrete(2)

        # -------------------------
        # OBSERVATION SPACE
        # -------------------------
        # state = (time step, current utility)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf], dtype=np.float32),
            high=np.array([T, np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.t = None
        self.u = None

        return

    #---------------
    # reset function
    #---------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.u = self._sample_apartment()

        obs = np.array([self.t, self.u], dtype=np.float32)
        info = {}

        return obs, info

    #--------------
    # step function
    #--------------
    def step(self, action: int):

        terminated = False
        truncated = False
        reward = 0.0

        # -------------------------
        # ACCEPT
        # -------------------------
        if action == 1:
            reward = float(self.u)
            terminated = True

        # -------------------------
        # REJECT
        # -------------------------
        elif action == 0:
            reward = -(self.u + self.t)

            self.t += 1

            # end of horizon → terminal with 0 reward
            if self.t >= self.T:
                reward = 0.0
                terminated = True
            else:
                self.u = self._sample_apartment()

        else:
            raise ValueError("Action must be 0 (reject) or 1 (accept)")

        obs = np.array([self.t, self.u], dtype=np.float32)
        info = {}

        return obs, reward, terminated, truncated, info
    
    #------------------
    # Transition kernel
    #------------------
    def _sample_apartment(self):
        return self.rng.integers(1, self.K + 1)
