# Jack Farrell
# Reinforcement Learning
# Problem Set 1
# Problem 3

'''Value Iteration Solution'''

import numpy as np

def policy_emergence(P, gamma, theta, check_policy):
    n_states = len(P)
    n_actions = len(P[0])

    policy = np.zeros(n_states, dtype=int)

    V = np.zeros(n_states)
    tolerance = theta * (1-gamma)/gamma

    backup = 0
    iteration = 0
    while True:

        V_new = np.zeros(n_states)

        for s in range(n_states):
            action_values = []

            # Calculate the reward for each action
            for a in range(n_actions):
                q_value = 0     # reset q_value to calculate the q_value for a state

                # Calculate the q_value for each action in a state and sum them to get the q_value for performing each action in that state
                for prob, s_next, reward, done in P[s][a]:
                    q_value += prob * (reward + gamma * V[s_next])

                action_values.append(q_value)
            
            V_new[s] = max(action_values)
      
        iteration += 1
    
        # Check Tolerance
        check_tolerance = np.max(np.abs(V_new - V))
        if check_tolerance < tolerance:
            break

        V = V_new


        # Compute π* after having found V*
        policy = np.zeros(n_states, dtype=int)

        for s in range(n_states):
            action_values = []

            for a in range(n_actions):
                q_value = 0

                for prob, s_next, reward, done in P[s][a]:
                    q_value += prob * (reward + gamma * V[s_next])

                action_values.append(q_value)

            policy[s] = np.argmax(action_values)


        # Create Grid
        policy_grid = policy.reshape(4, 4)
        V_grid = V.reshape(4,4)

        # Check policy
        if np.array_equal(policy, check_policy.flatten()):
            break

    
    return iteration, check_tolerance, policy_grid