import numpy as np
import matplotlib.pyplot as plt

from env import ApartmentEnv
from policies import RandomPolicy, ThresholdPolicy, OptimalPolicy


# ---------------------
# Run single episode
# ---------------------
def run_episode(env, policy):
    reset_out = env.reset()

    # -------------------------
    # SAFETY CHECK (IMPORTANT)
    # -------------------------
    if reset_out is None:
        raise ValueError(
            "env.reset() returned None. "
            "Your ApartmentEnv.reset() is missing a return (obs, info)."
        )

    obs, _ = reset_out
    done = False

    total_reward = 0.0
    rejected_everything = True

    while not done:
        action = policy.act(obs)

        if action == 1:
            rejected_everything = False

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward, rejected_everything


# ----------------
# Evaluation loop
# ----------------
def evaluate(policy, env, N=10000):
    returns = []
    reject_all = 0

    for _ in range(N):
        r, rej = run_episode(env, policy)
        returns.append(r)
        reject_all += int(rej)

    returns = np.array(returns)

    mean = np.mean(returns)
    se = np.std(returns) / np.sqrt(N)
    reject_frac = reject_all / N

    return mean, se, reject_frac, returns


# -----
# MAIN
# -----
def main():

    N = 10000
    T = 4
    K = 4

    env = ApartmentEnv(T=T, K=K, seed=42)

    results = {}

    # -------------------------
    # Random policy
    # -------------------------
    print("Running Random Policy...")
    rand_policy = RandomPolicy(T=T)
    results["Random"] = evaluate(rand_policy, env, N)

    # -------------------------
    # Threshold sweep
    # -------------------------
    best_u = None
    best_score = -np.inf
    threshold_results = {}

    for u_min in [1, 2, 3, 4]:
        print(f"Running ThresholdPolicy u_min={u_min}...")
        policy = ThresholdPolicy(u_min)

        mean, se, rej, returns = evaluate(policy, env, N)
        threshold_results[u_min] = (mean, se, rej, returns)

        if mean > best_score:
            best_score = mean
            best_u = u_min

    # -------------------------
    # Optimal policy
    # -------------------------
    print("Running Optimal Policy...")
    opt_policy = OptimalPolicy(T=T, K=K)
    results["Optimal"] = evaluate(opt_policy, env, N)

    # -------------------------
    # Print results
    # -------------------------
    print("\n================ RESULTS ================\n")

    for name, (mean, se, rej, _) in results.items():
        print(f"{name}:")
        print(f"  Mean return = {mean:.4f}")
        print(f"  Std error   = {se:.4f}")
        print(f"  Reject-all fraction = {rej:.4f}\n")

    print("Threshold sweep results:")
    for u_min, (mean, se, rej, _) in threshold_results.items():
        print(f"u_min={u_min}: mean={mean:.4f}, se={se:.4f}")

    print(f"\nBEST THRESHOLD: u_min = {best_u}\n")

    # -------------------------
    # Plot histograms
    # -------------------------
    plt.figure(figsize=(10, 6))
    bins = np.linspace(-10, 10, 50)

    plt.hist(results["Random"][3], bins=bins, alpha=0.4, label="Random")
    plt.hist(results["Optimal"][3], bins=bins, alpha=0.4, label="Optimal")

    best_returns = threshold_results[best_u][3]
    plt.hist(best_returns, bins=bins, alpha=0.4,
             label=f"Threshold (u={best_u})")

    plt.title("Return Distributions (T=4, K=4)")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()