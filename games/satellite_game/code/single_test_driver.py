import pickle
import time

import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import MainConfig
from evaluation import (
    compute_episode_summary,
    compute_value_functions,
    evaluate_ADP_policy,
    evaluate_AMG_policy,
    evaluate_boltzmann_policy,
    evaluate_FP_policy,
    evaluate_H2S_policy,
    evaluate_MC_policy,
    evaluate_reduced_k_boltzmann_policy,
    generate_all_trajectories,
    print_episode_summary,
)
from plotting import (
    plot_all_latent_state_trajectories,
    plot_latent_and_probs,
    plot_policy_action_heatmaps,
)

output_prefix = "single_evaluation_output"  # <-- output name prefix

# === MAIN SCRIPT ===

seed = 4

np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
config = MainConfig.create(subkey)

print(config)

results_all = []
summaries = []

# --- Boltzmann ---
np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
results = evaluate_reduced_k_boltzmann_policy(key, seed, config)
summary = compute_episode_summary(results, config)
print_episode_summary(summary)
results_all.append(results)
summaries.append(summary)

# --- H2S (example H=1) ---
np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
H1_trajectories = generate_all_trajectories(H=1)
results = evaluate_H2S_policy(key, seed, config, H=1, trajectories=H1_trajectories)
summary = compute_episode_summary(results, config)
print_episode_summary(summary)
results_all.append(results)
summaries.append(summary)

# --- AMG ---
np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
results = evaluate_AMG_policy(key, seed, config)
summary = compute_episode_summary(results, config)
print_episode_summary(summary)
results_all.append(results)
summaries.append(summary)

# --- ADP ---
np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
model_matrix = compute_value_functions(subkey, config, config.game.num_timesteps)
results = evaluate_ADP_policy(key, seed, config, model_matrix)
summary = compute_episode_summary(results, config)
print_episode_summary(summary)
results_all.append(results)
summaries.append(summary)

# --- MC ---
np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
results = evaluate_MC_policy(key, seed, config)
summary = compute_episode_summary(results, config)
print_episode_summary(summary)
results_all.append(results)
summaries.append(summary)

# --- FP ---
np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
results = evaluate_FP_policy(key, seed, config)
summary = compute_episode_summary(results, config)
print_episode_summary(summary)
results_all.append(results)
summaries.append(summary)

res = results_all[0]  # e.g. Boltzmann

custom_names = {
    "Boltzmann_Reduced_K": "Boltzmann",
    "H2S (H=1)": "H2S (Horizon=1)",
    "AMG": "AMG",
    "ADP": "ADP",
    "MC": "MC",
    "FP": "FP",
}

action_labels = {0: "Intercept", 1: "Hold Position", 2: "Retreat"}

line_style_map = {
    "Boltzmann_Reduced_K": {"color": "orange", "linestyle": "-", "linewidth": 2},
    "H2S (H=1)": {"color": "green", "linestyle": "--", "linewidth": 2},
    "AMG": {"color": "blue", "linestyle": ":", "linewidth": 2},
    "ADP": {"color": "purple", "linestyle": "-.", "linewidth": 2},
    "MC": {"color": "black", "linestyle": "-", "linewidth": 3},
    "FP": {"color": "red", "linestyle": "--", "linewidth": 2},
}

plot_latent_and_probs(res, save_prefix="plot")
plot_all_latent_state_trajectories(
    results_all, summaries, rename_policies=custom_names, style_map=line_style_map
)
plot_policy_action_heatmaps(
    results_all, rename_policies=custom_names, action_labels=action_labels
)

# === SAVE OUTPUT ===
with open(f"{output_prefix}_results.pkl", "wb") as f:
    pickle.dump(results_all, f)
with open(f"{output_prefix}_summaries.pkl", "wb") as f:
    pickle.dump(summaries, f)

# # Save filter data as CSV for plotting
# for res in results_all:
#     policy = res["policy"].replace(" ", "_")
#     np.savetxt(f"{output_prefix}_{policy}_filter_means.csv", np.array(res["filter_means"]), delimiter=",")
#     np.savetxt(f"{output_prefix}_{policy}_filter_stds.csv", np.array(res["filter_stds"]), delimiter=",")
#     np.savetxt(f"{output_prefix}_{policy}_filter_p5.csv", np.array(res["filter_p5"]), delimiter=",")
#     np.savetxt(f"{output_prefix}_{policy}_filter_p95.csv", np.array(res["filter_p95"]), delimiter=",")
