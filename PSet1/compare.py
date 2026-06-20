# Jack Farrell
# Reinforcement Learning
# Problem Set 1
# Problem 3

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

from vi import value_iteration
from pi import policy_iteration
from policy_emergence import policy_emergence

env = gym.make("FrozenLake-v1", is_slippery=True)
P = env.unwrapped.P   # list of (prob, next_state, reward, terminated)

# Value Iteration Policy Solution
print("------------------------")
print("Part A: Value Iteration")
print("------------------------")

V_vi, policy_vi, iteration_vi, backup_vi = value_iteration(P, gamma=0.99, theta=10**-4)

arrows = np.array(["←", "↓", "→", "↑"])     # 0="←", 1="↓", 2="→", 3="↑"
policy_grid_vi = arrows[policy_vi].reshape(4, 4)
np.set_printoptions(suppress=True, precision=3)
print("Optimal Policy found on iteration ", iteration_vi, "\n")
print("V*: \n", V_vi, "\n")
print("π*: \n", policy_grid_vi, "\n")
print("Policy and Reward Grid: ")
for i in range(4):
    for j in range(4):
        print(f"{policy_vi[i,j]}({V_vi[i,j]:.2f})", end="  ")
    print()


# Policy Iteration Solution
print("-------------------------")
print("Part B: Policy Iteration")
print("-------------------------")

V_pi, policy_pi, iteration_pi, backup_pi = policy_iteration(P, gamma=0.99)

arrows = np.array(["←", "↓", "→", "↑"])     # 0="←", 1="↓", 2="→", 3="↑"
policy_grid_pi = arrows[policy_pi].reshape(4, 4)
print("Optimal Policy found on iteration ", iteration_pi, "\n")
print("V*: \n", V_pi, "\n")
print("π*: \n", policy_grid_pi, "\n")
print("Policy and Reward Grid: ")
for i in range(4):
    for j in range(4):
        print(f"{policy_pi[i,j]}({V_vi[i,j]:.2f})", end="  ")
    print()


# Empirical Comparison Solution
print("-----------------------------")
print("Part C: Empirical Comparison")
print("-----------------------------")

gamma_list = [0.5, 0.9, 0.99, 0.999]

# Value Iteration
print("Value Iteration Comparison for Various Gammas\n")
print(f"{'γ':<6} | {'Iterations':<10} | {'Wall Time (s)':<14} | {'Backups':<10}")
print("-" * 55)

VI_iterations = []
VI_time = []
for gamma in gamma_list:
    start = time.perf_counter()
    V_vi, policy_vi, iteration_vi, backup_vi = value_iteration(P, gamma, theta=10**-4)
    stop = time.perf_counter()

    VI_iterations.append(iteration_vi)
    VI_time.append(stop-start)

    print(f"{gamma:<6} | {iteration_vi:<10} | {stop-start:<14.6f} | {backup_vi:<10}")

# Policy Iteration
print("\n\nPolicy Iteration Comparison for Various Gammas\n")
print(f"{'γ':<6} | {'Iterations':<10} | {'Wall Time (s)':<14} | {'Backups':<10}")
print("-" * 55)

PI_iterations = []
PI_time = []
for gamma in gamma_list:
    start = time.perf_counter()
    V_pi, policy_pi, iteration_pi, backup_pi = policy_iteration(P, gamma)
    stop = time.perf_counter()

    PI_iterations.append(iteration_pi)
    PI_time.append(stop-start)

    print(f"{gamma:<6} | {iteration_pi:<10} | {stop-start:<14.6f} | {backup_pi:<10}")

# Plot VI vs. PI Iteration count
plt.figure()
plt.title("Iterations for Various γ Values")
plt.plot(gamma_list, VI_iterations, marker='o', label="Value Iteration")
plt.plot(gamma_list, PI_iterations, marker='o', label="Policy Iteration")
plt.xlabel("γ")
plt.ylabel("Iterations")
plt.grid()
plt.legend()


plt.show()


print("-------------------------")
print("Part D: Policy Emergence")
print("-------------------------")

k_star, check_tolerance, policy_output = policy_emergence(P, gamma=0.99, theta=10**-4, check_policy=policy_vi)
print("Iteration: ", k_star)
print("Tolerance: ", check_tolerance)