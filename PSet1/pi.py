# Jack Farrell
# Reinforcement Learning
# Problem Set 1
# Problem 3

'''Policy Iteration Solution'''

import numpy as np


def policy_iteration(P, gamma):

    n_states = len(P)
    n_actions = len(P[0])
    policy = np.zeros(n_states, dtype=int)
    iteration = 0
    backup = 0

    # -----------------------
    # Policy evaluation step
    # -----------------------
    def policy_evaluation_matrix(P, gamma, policy):
        n_states = len(P)
        P_pi = np.zeros((n_states, n_states))
        R_pi = np.zeros(n_states)

        for s in range(n_states):
            a = policy[s]

            # Build P_pi and R_pi matrices
            for prob, s_next, reward, done in P[s][a]:
                P_pi[s, s_next] += prob
                R_pi[s] += prob * reward

        I = np.eye(n_states)

        V = np.linalg.solve(I - gamma * P_pi, R_pi)
        return V

    # ----------------------
    # Policy iteration step
    #-----------------------

    while True:
        old_policy = policy.copy()

        # Evaluate the policy
        V = policy_evaluation_matrix(P, gamma, policy)

        # Policy improvement
        for s in range(n_states):
            action_values = []
            
            for a in range(n_actions):
                q = 0   # resets q for eact action evaluated

                for prob, s_next, reward, done in P[s][a]:
                    q += prob * (reward + gamma * V[s_next])
                
                action_values.append(q)

            # Prevents oscillation of policy where two or more actions may be best
            max_q = np.max(action_values)
            best_actions = np.where(np.isclose(action_values, max_q))[0]
            policy[s] = best_actions[0]

        #Check policy
        if np.array_equal(policy, old_policy):
            break

        iteration += 1

    policy_grid = policy.reshape(4, 4)
    V_grid = V.reshape(4,4)

    # Calculate Bellman Backups
    backup = n_states**3 * iteration
    
    return V_grid, policy_grid, iteration, backup



