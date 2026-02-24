import os
import time
import pandas as pd
import filters as jf
import jax
import jax.numpy as jnp
import models
import numpy as np
import matplotlib.pyplot as plt
import plotting as plotting
from config import MainConfig
from benchmark_policies import get_MC_action_jax
from policies import (
    generate_all_trajectories,
    get_means,
    get_H2S_action
)
import math
from filters import print_particle_with_labels
from utility_functions import utility, utility_debugger
import constants as C
from constants import (
    LOGICAL_PARTICLE_DIM
)
from ADP import compute_value_functions, get_ADP_action_MLP, get_value_estimate_with_model, generate_cost_function_estimate
import regressions as rf
from game_managers import ADSGameManager
from constants import HIDDEN_DIMS, IDX_XD1, IDX_XD2
from ADP_debugger import print_action_values, debug_ADP_output, print_action_decomp

seed = 4

np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
config = MainConfig.create(subkey)

# Start the EWA model
model = models.MannedVehicle(config)

# Start up a game manager
manager = ADSGameManager(config)

# Intialize particles

start_time = time.perf_counter()
key, subkey = jax.random.split(key)
num_particles = config.filter.num_particles
num_attacker_actions = config.game.num_attacker_actions
num_defender_actions = int(config.game.num_defender_actions)
particles = jf.initialize_particles_beta_prior(
    subkey, config, num_particles, num_attacker_actions
)
end_time = time.perf_counter()

print(f"Time to Initialize Particles: {end_time - start_time:.6f} seconds")

print_particle_with_labels(get_means(particles), config)

num_steps = config.game.num_timesteps

time_per_step = []
filter_data = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # D = number of dimensions in your particles
filter_stds = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # same shape as means
filter_mins = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # D = number of dimensions in your particles
filter_maxs = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # D = number of dimensions in your particles

H=2

trajectories = generate_all_trajectories(H, config)

records = []

# input_dimensions = LOGICAL_PARTICLE_DIM + config.game.num_defender_actions #One hot encoding
input_dimensions = LOGICAL_PARTICLE_DIM + 2
path = "/home/chris/Desktop/bayesian-security-games/games/ADS_merging/ADP_models/mlp_sequence_TEST.msgpack"


# # Generate models
# print("Training ADP Models...")
# key, subkey = jax.random.split(key)
# model_seq, debug_data = compute_value_functions(subkey, config, num_steps, num_epochs=1000)

# # # Save models
# rf.save_params_sequence(model_seq, path)

#Load models
model_seq = rf.load_state_sequence(
    path,
    num_steps,
    input_dim=input_dimensions,
    hidden_dims=HIDDEN_DIMS,
    rng=jax.random.PRNGKey(0),
)



defender_actions = jnp.arange(num_defender_actions)

defender_coords = [] #Stores x,y positions
attacker_coords = []

# Unpack defender & attacker observations (assuming 4D each: [x, y, heading, speed])
XD1_obs, XD2_obs, XD3_obs, XD4_obs, XA1_obs, XA2_obs, XA3_obs, XA4_obs = manager.state

defender_coords.append([XD1_obs, XD2_obs])
attacker_coords.append([XA1_obs, XA2_obs])

POLICY = "ADP"
# POLICY = "H2S"
# POLICY = "MC"

start = time.time()
for i in range(num_steps):

    start_time = time.perf_counter()

    mean_particle = get_means(particles)

    # The first step is to get the actions
    key, subkey = jax.random.split(key)

    if POLICY == "ADP":
        defender_action = get_ADP_action_MLP(mean_particle, model_seq[i], defender_actions, config)
    elif POLICY == "H2S":
        defender_action = get_H2S_action(subkey, particles, trajectories, config)   
    else:
        defender_action = get_MC_action_jax(model.probabilities, manager.state, config)

    attacker_action = model.select_action()
    # attacker_action = 4
    actions = jnp.array([attacker_action, defender_action])
    end_time = time.perf_counter()

    # A round of the game is then played
    defender_observation, attacker_observation = manager.step(
        attacker_action, defender_action
    )

    # The opponent's model is then updated
    model.update_model(attacker_observation, defender_action, config)

    # Finally, we update our particle filter
    key, subkey = jax.random.split(key)
    start_time_filter = time.perf_counter()
    particles = jf.step(
        subkey,
        particles,
        attacker_action,
        defender_action,
        defender_observation,
        config,
    )
    end_time_filter = time.perf_counter()

    mean_particle = get_means(particles)
    std_particle = jf.get_stds(particles)
    min_particle = jf.get_mins(particles)
    max_particle = jf.get_maxs(particles)

    print(jnp.std(particles[:, IDX_XD1]))
    print(jnp.std(particles[:, IDX_XD2]))

    # print_particle_with_labels(min_particle, config)
    # print_particle_with_labels(max_particle, config)

    filter_data[i] = np.array(mean_particle)
    filter_stds[i] = np.array(std_particle)
    filter_mins[i] = np.array(min_particle)
    filter_maxs[i] = np.array(max_particle)

    
    # Record Truth
    XD1_obs, XD2_obs, XD3_obs, XD4_obs, XA1_obs, XA2_obs, XA3_obs, XA4_obs = manager.state

    truth_particle = jnp.array([
        model.experience,
        config.model.delta,
        config.model.phi,
        config.model.rho,
        config.model.Lambda,
        model.beta_A1,
        model.beta_A2,
        model.beta_A3,
        model.beta_A4,
        model.beta_A5,
        model.beta_A6,
        config.game.error_Z_sigma,
        XD1_obs,
        XD2_obs,
        XD3_obs,
        XD4_obs,
        XA1_obs,
        XA2_obs,
        XA3_obs,
        XA4_obs,
    ])

    truth_particle = jnp.hstack([truth_particle, model.attractions, model.probabilities])

    defender_coords.append([XD1_obs, XD2_obs])
    attacker_coords.append([XA1_obs, XA2_obs])

    # Utilities (if these are your current functions)
    atk_util = utility(
        XA2_obs,               # attacker's true y
        XA1_obs,
        XA2_obs,
        XA3_obs,
        XA4_obs,
        XD1_obs,
        XD2_obs,
        XD3_obs,
        XD4_obs,
        attacker_action,
        config.model.beta_A1,
        config.model.beta_A2,
        config.model.beta_A3,
        config.model.beta_A4,
        config.model.beta_A5,
        config.model.beta_A6,
        config,
        is_defender=False
    )

    def_util= utility(
        XD2_obs,               # defender's true y
        XD1_obs,
        XD2_obs,
        XD3_obs,
        XD4_obs,
        XA1_obs,
        XA2_obs,
        XA3_obs,
        XA4_obs,
        defender_action,
        config.game.defender_beta_1,
        config.game.defender_beta_2,
        config.game.defender_beta_3,
        config.game.defender_beta_4,
        config.game.defender_beta_5,
        config.game.defender_beta_6,
        config,
        is_defender=True
)

    # Action unpack (accel, steer)
    def_acc, def_steer = config.game.defender_action_set[defender_action]
    att_acc, att_steer = config.game.attacker_action_set[attacker_action]

    print(
        f"t={i:02d} | "
        f"D[a={def_acc:5.2f}, δ={def_steer:5.1f}] "
        f"A[a={att_acc:5.2f}, δ={att_steer:5.1f}] | "
        f"D(x={XD1_obs:7.2f}, y={XD2_obs:7.2f}, steer={XD3_obs:5.2f}, v={XD4_obs:5.2f}) | "
        f"A(x={XA1_obs:7.2f}, y={XA2_obs:7.2f}, steer={XD3_obs:5.2f}, v={XA4_obs:5.2f}) | "
        f"U_D={def_util:7.3f}, U_A={atk_util:7.3f} | "
        # f"Offroad_Penalty={off_road:7.3f}, Speeding={speed_violation_penalty:7.3f}, Proximity={car_proximity_penalty:7.3f}, Nav={navigation_utility:7.3f} | "
        # f"t_act={end_time - start_time:7.4f}s, t_filt={end_time_filter - start_time_filter:7.4f}s"
    )

    # ================= Credible intervals for all dimensions =================

    particle_dim = LOGICAL_PARTICLE_DIM
    quantiles = jnp.array([5.0, 25.0, 50.0, 75.0, 95.0])

    # Compute quantiles: shape (5, particle_dim)
    q = jnp.percentile(particles[:, :particle_dim], quantiles, axis=0)

    mean_est = jnp.mean(particles[:, :particle_dim], axis=0)

    # Convert to numpy for storage
    q = np.asarray(q)
    mean_est = np.asarray(mean_est)
    truth = np.asarray(truth_particle[:particle_dim])

    for d in range(particle_dim):
        records.append({
            "t": i,
            "dim": d,
            "truth": truth[d],
            "mean": mean_est[d],
            "q05": q[0, d],
            "q25": q[1, d],
            "q50": q[2, d],
            "q75": q[3, d],
            "q95": q[4, d],
        })


defender_coords = np.array(defender_coords)
attacker_coords = np.array(attacker_coords)
timesteps = np.arange(len(defender_coords))

df_traj = pd.DataFrame({
    "t": timesteps,

    "def_x": defender_coords[:, 0],
    "def_y": defender_coords[:, 1],

    "att_x": attacker_coords[:, 0],
    "att_y": attacker_coords[:, 1],
})

# df_traj.to_csv(f"merge_game_trajectories_{POLICY}.csv", index=False)



# Lane and merge boundaries for context
l1, l2, l3 = config.game.l1, config.game.l2, config.game.l3
W1, W2 = config.game.W1, config.game.W2

# Trajectory Plot #1

plt.figure(figsize=(8, 6))

# ================= Extract parameters =================
l1 = config.game.l1
l2 = config.game.l2
l3 = config.game.l3
W1 = config.game.W1
W2 = config.game.W2

# ================= Trajectory bounds =================
y_min = min(defender_coords[:, 1].min(), attacker_coords[:, 1].min(), 0.0)
y_max = max(defender_coords[:, 1].max(), attacker_coords[:, 1].max(), W2)

y_vals = np.linspace(y_min, y_max, 400)

# ================= True tapering right boundary =================
merge_progress = (y_vals - W1) / (W2 - W1)
merge_progress = np.clip(merge_progress, 0.0, 1.0)

right_edge = (1.0 - merge_progress) * l3 + merge_progress * l2

# ================= Road shading =================
# Full road before taper
plt.fill_betweenx(
    y_vals,
    l1,
    right_edge,
    color="gray",
    alpha=0.35,
    zorder=0,
    label="Drivable road"
)

# Disappearing right lane (visual emphasis)
plt.fill_betweenx(
    y_vals,
    right_edge,
    l3,
    color="lightgray",
    alpha=0.9,
    zorder=1,
    label="Lane removed after merge"
)

# ================= Road boundaries =================
# Left road edge
plt.axvline(l1, color="black", linewidth=3)

# Centerline
plt.axvline(l2, color="black", linestyle="--", linewidth=2)

# Tapering right edge
plt.plot(
    right_edge,
    y_vals,
    color="black",
    linewidth=3,
    label="Right road edge (tapering)"
)

# ================= Merge region lines =================
plt.axhline(W1, color="tab:blue", linestyle="--", linewidth=2)
plt.axhline(W2, color="tab:red", linestyle="-.", linewidth=2)

plt.text(l3 + 0.1, W1, "Soft merge (W1)", color="tab:blue", va="top")
plt.text(l3 + 0.1, W2, "Hard merge (W2)", color="tab:red", va="top")

# ================= Trajectories =================
plt.plot(
    defender_coords[:, 0],
    defender_coords[:, 1],
    marker="o",
    linestyle="-",
    linewidth=2,
    color="tab:blue",
    label="Defender",
    zorder=5,
)

plt.plot(
    attacker_coords[:, 0],
    attacker_coords[:, 1],
    marker="x",
    linestyle="--",
    linewidth=2,
    color="tab:orange",
    label="Attacker",
    zorder=5,
)

# ================= Formatting =================
plt.xlabel("Horizontal position (m)")
plt.ylabel("Vertical position (m)")
plt.title("Merge Game Trajectories with Tapering Road Geometry")

plt.xlim(l1 - 0.5, l3 + 1.5)
plt.ylim(y_min - 1.0, y_max + 1.0)

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("car_trajectories_with_taper.png", dpi=200)
plt.show()

# Trajectory Plot #2

plt.figure(figsize=(8, 6))

# ================= Extract parameters =================
l1 = config.game.l1
l2 = config.game.l2
l3 = config.game.l3
W1 = config.game.W1
W2 = config.game.W2

# ================= Trajectory bounds =================
y_min = min(defender_coords[:, 1].min(), attacker_coords[:, 1].min(), 0.0)
y_max = max(defender_coords[:, 1].max(), attacker_coords[:, 1].max(), W2)

y_vals = np.linspace(y_min, y_max, 400)

# ================= True tapering right boundary =================
merge_progress = (y_vals - W1) / (W2 - W1)
merge_progress = np.clip(merge_progress, 0.0, 1.0)

right_edge = (1.0 - merge_progress) * l3 + merge_progress * l2

# ================= Road shading =================
# Full road before taper
plt.fill_betweenx(
    y_vals,
    l1,
    right_edge,
    color="gray",
    alpha=0.35,
    zorder=0,
    label="Drivable road"
)

# Disappearing right lane (visual emphasis)
plt.fill_betweenx(
    y_vals,
    right_edge,
    l3,
    color="lightgray",
    alpha=0.9,
    zorder=1,
    label="Lane removed after merge"
)

# ================= Road boundaries =================
# Left road edge
plt.axvline(l1, color="black", linewidth=3)

# Centerline
plt.axvline(l2, color="black", linestyle="--", linewidth=2)

# Tapering right edge
plt.plot(
    right_edge,
    y_vals,
    color="black",
    linewidth=3,
    label="Right road edge (tapering)"
)

# ================= Merge region lines =================
plt.axhline(W1, color="tab:blue", linestyle="--", linewidth=2)
plt.axhline(W2, color="tab:red", linestyle="-.", linewidth=2)

plt.text(l3 + 0.1, W1, "Soft merge (W1)", color="tab:blue", va="top")
plt.text(l3 + 0.1, W2, "Hard merge (W2)", color="tab:red", va="top")

# ================= Trajectories =================
plt.plot(
    defender_coords[:, 0],
    defender_coords[:, 1],
    marker="o",
    linestyle="-",
    linewidth=2,
    color="tab:blue",
    label="Defender",
    zorder=5,
)

plt.plot(
    attacker_coords[:, 0],
    attacker_coords[:, 1],
    marker="x",
    linestyle="--",
    linewidth=2,
    color="tab:orange",
    label="Attacker",
    zorder=5,
)

# ================= Formatting =================
plt.xlabel("Horizontal position (m)")
plt.ylabel("Vertical position (m)")
plt.title("Merge Game Trajectories with Tapering Road Geometry")

plt.xlim(l1 - 0.5, l3 + 1.5)
plt.ylim(y_min - 1.0, y_max + 1.0)

# 1:1 scale
plt.gca().set_aspect(0.5, adjustable="box")


plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0.0,
    frameon=True
)


plt.tight_layout()
plt.savefig("car_trajectories_with_taper_1to2.png", dpi=200)


timesteps = np.arange(len(defender_coords))

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --------------------------------------------------
# Horizontal position vs time
# --------------------------------------------------
axes[0].plot(
    timesteps, defender_coords[:, 0],
    marker='o', linestyle='-', label='Defender'
)
axes[0].plot(
    timesteps, attacker_coords[:, 0],
    marker='x', linestyle='--', label='Attacker'
)

# Lane boundaries
axes[0].axhline(l1, color='k', linestyle=':', linewidth=1)
axes[0].axhline(l2, color='k', linestyle='--', linewidth=1)
axes[0].axhline(l3, color='k', linestyle=':', linewidth=1)

axes[0].set_ylabel("Horizontal position (m)")
axes[0].set_title("Horizontal Position vs Time")
axes[0].legend()
axes[0].grid(True)

# --------------------------------------------------
# Vertical position vs time
# --------------------------------------------------
axes[1].plot(
    timesteps, defender_coords[:, 1],
    marker='o', linestyle='-', label='Defender'
)
axes[1].plot(
    timesteps, attacker_coords[:, 1],
    marker='x', linestyle='--', label='Attacker'
)

# Merge region thresholds
axes[1].axhline(W1, color='gray', linestyle='--', linewidth=1, label='Soft merge (W1)')
axes[1].axhline(W2, color='gray', linestyle='-.', linewidth=1, label='Hard merge (W2)')

axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Vertical position (m)")
axes[1].set_title("Vertical Position vs Time")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("car_positions_vs_time.png")

df_ci = pd.DataFrame(records)


particle_dim = LOGICAL_PARTICLE_DIM

n_cols = 3                          # good default
n_rows = math.ceil(particle_dim / n_cols)

dim_names = {
    # ================= Experience / Learning =================
    C.IDX_N:        "Experience N",

    # ================= EWA Parameters =================
    C.IDX_DELTA:   "EWA δ (delta)",
    C.IDX_PHI:     "EWA φ (phi)",
    C.IDX_RHO:     "EWA ρ (rho)",
    C.IDX_LAMBDA:  "EWA λ (lambda)",

    # ================= Attacker Payoff Parameters =================
    C.IDX_BETA_A1: "Attacker β₁",
    C.IDX_BETA_A2: "Attacker β₂",
    C.IDX_BETA_A3: "Attacker β₃",
    C.IDX_BETA_A4: "Attacker β₄",
    C.IDX_BETA_A5: "Attacker β₅",
    C.IDX_BETA_A6: "Attacker β₆",

    # ================= Attacker Noise =================
    C.IDX_SIGMA:   "Attacker σ (noise)",

    # ================= Defender State =================
    C.IDX_XD1:     "Defender x-position (m)",
    C.IDX_XD2:     "Defender y-position (m)",
    C.IDX_XD3:     "Defender heading (deg)",
    C.IDX_XD4:     "Defender speed (m/s)",

    # ================= Attacker State =================
    C.IDX_XA1:     "Attacker x-position (m)",
    C.IDX_XA2:     "Attacker y-position (m)",
    C.IDX_XA3:     "Attacker heading (deg)",
    C.IDX_XA4:     "Attacker speed (m/s)",

    # ================= EWA Attractions =================
    C.IDX_A1 + 0:  "Attraction A₁",
    C.IDX_A1 + 1:  "Attraction A₂",
    C.IDX_A1 + 2:  "Attraction A₃",
    C.IDX_A1 + 3:  "Attraction A₄",
    C.IDX_A1 + 4:  "Attraction A₅",
    C.IDX_A1 + 5:  "Attraction A₆",
    C.IDX_A1 + 6:  "Attraction A₇",
    C.IDX_A1 + 7:  "Attraction A₈",
    C.IDX_A1 + 8:  "Attraction A₉",

    # ================= Action Probabilities =================
    C.IDX_P1 + 0:  "Action prob P₁",
    C.IDX_P1 + 1:  "Action prob P₂",
    C.IDX_P1 + 2:  "Action prob P₃",
    C.IDX_P1 + 3:  "Action prob P₄",
    C.IDX_P1 + 4:  "Action prob P₅",
    C.IDX_P1 + 5:  "Action prob P₆",
    C.IDX_P1 + 6:  "Action prob P₇",
    C.IDX_P1 + 7:  "Action prob P₈",
    C.IDX_P1 + 8:  "Action prob P₉",
}


fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(5 * n_cols, 3 * n_rows),
    sharex=True,
)

axes = axes.flatten()

for d in range(particle_dim):
    ax = axes[d]
    sub = df_ci[df_ci["dim"] == d]

    # 90% credible interval
    ax.fill_between(
        sub["t"],
        sub["q05"],
        sub["q95"],
        alpha=0.25,
        label="90% CI" if d == 0 else None,
    )

    # 50% credible interval
    ax.fill_between(
        sub["t"],
        sub["q25"],
        sub["q75"],
        alpha=0.45,
        label="50% CI" if d == 0 else None,
    )

    # Particle mean
    ax.plot(
        sub["t"],
        sub["mean"],
        linewidth=2,
        label="Mean" if d == 0 else None,
    )

    # Ground truth
    ax.plot(
        sub["t"],
        sub["truth"],
        linestyle="--",
        color="black",
        linewidth=1.5,
        label="Truth" if d == 0 else None,
    )

    ax.set_title(dim_names.get(d, f"State dim {d}"))
    ax.grid(True, linestyle=":", alpha=0.6)

# Remove unused axes
for d in range(particle_dim, len(axes)):
    fig.delaxes(axes[d])

# Global labels
fig.supxlabel("Timestep")
fig.supylabel("State value")

# Shared legend (only once)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=4,
    frameon=False,
)

fig.suptitle("Particle Filter Credible Intervals (All Dimensions)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("filter_plot.png")



# jf.print_particle_with_labels(jf.get_means(particles))

np.savetxt("filter_output_last.csv", particles, delimiter=",")


def consolidate_model_data(
    latent_state_record,
    experience_record,
    attraction_record,
    probability_record,
    config: MainConfig,
):
    """
    Consolidates model data into a single matrix.
    Each row corresponds to a timestep.
    Columns are ordered as follows:
    [delta, phi, rho, lambda, N, beta_1, beta_2, beta_3, X, eta,
     A0, A1, A2, P0, P1, P2]

    Args:
        latent_state_record (list): list of X over timesteps
        eta (float): constant eta value
        EWA_params (dict): parameters dict with keys: delta, phi, rho, lambda, beta_1, beta_2, beta_3
        experience_record (list): list of N over timesteps
        attraction_record (list of lists): list of [A0,A1,A2] over timesteps
        probability_record (list of lists): list of [P0,P1,P2] over timesteps

    Returns:
        np.ndarray: matrix of shape (n_timesteps, 16)
    """
    # number of timesteps
    n_timesteps = len(latent_state_record)

    # extract EWA parameters in correct order
    params = np.array(
        [
            config.model.delta,
            config.model.phi,
            config.model.rho,
            config.model.Lambda,
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    # replicate params & eta across all timesteps
    params_tiled = np.tile(params, (n_timesteps, 1))
    eta_column = np.full(
        (n_timesteps, 1),
        config.game.error_Z_variance,
    )

    # convert per-timestep records to arrays
    experience = np.array(experience_record).reshape(-1, 1)
    latent_state = np.array(latent_state_record).reshape(-1, 1)
    attractions = np.array(attraction_record)  # shape: (n,3)
    probabilities = np.array(probability_record)  # shape: (n,3)

    # construct the final matrix column by column
    matrix = np.hstack(
        [
            params_tiled[:, 0:1],  # delta
            params_tiled[:, 1:2],  # phi
            params_tiled[:, 2:3],  # rho
            params_tiled[:, 3:4],  # lambda
            experience,  # N
            params_tiled[:, 4:5],  # beta_1
            params_tiled[:, 5:6],  # beta_2
            params_tiled[:, 6:7],  # beta_3
            params_tiled[:, 7:8],  # beta_3
            latent_state,  # X
            eta_column,  # eta
            attractions[:, 0:1],  # A0
            attractions[:, 1:2],  # A1
            attractions[:, 2:3],  # A2
            probabilities[:, 0:1],  # P0
            probabilities[:, 1:2],  # P1
            probabilities[:, 2:3],  # P2
        ]
    )

    return matrix[:-1, :]


end = time.time()

print(f"Time Elapsed: {end - start}")


df = pd.DataFrame(records)
df.to_csv("game_log.csv", index=False)

# # Plot the true and filter data of the EWA model

# plotting.plot_means_over_time(model_data, filter_data, filter_stds)
# plotting.plot_latent_and_probs(model_data, filter_data, filter_stds)
# # plotting.plot_means_over_time(model_data, filter_mins)
# # plotting.plot_means_over_time(model_data, filter_maxs)
# # plotting.plot_differences_over_time(model_data, filter_data)
