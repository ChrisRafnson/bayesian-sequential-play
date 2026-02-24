import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import BoundaryNorm, ListedColormap

IDX_DELTA = 0
IDX_PHI = 1
IDX_RHO = 2
IDX_LAMBDA = 3
IDX_N = 4
IDX_BETA_1 = 5
IDX_BETA_2 = 6
IDX_BETA_3 = 7
IDX_BETA_4 = 8
IDX_X = 9
IDX_ETA = 10
IDX_A0 = 11
IDX_A1 = 12
IDX_A2 = 13
IDX_P0 = 14
IDX_P1 = 15
IDX_P2 = 16

PARAMETER_INDICES = jnp.array(
    [IDX_DELTA, IDX_PHI, IDX_RHO, IDX_LAMBDA, IDX_N]
)  # Tells JAX where to update the parameters, delta, phi, rho, lambda, and the experience
ATTRACTION_INDICES = jnp.array(
    [IDX_A0, IDX_A1, IDX_A2]
)  # Tells JAX where to update the attractions
PROBABILITY_INDICES = jnp.array(
    [IDX_P0, IDX_P1, IDX_P2]
)  # Tells JAX where to update the probabilities
BETA_INDICES = jnp.array(
    [IDX_BETA_1, IDX_BETA_2, IDX_BETA_3, IDX_BETA_4]
)  # Tells JAX where to update the betas
X_INDICES = jnp.array([IDX_X])  # Tells JAX where to update the latent state
ETA_INDICES = jnp.array([IDX_ETA])  # Tells JAX where to update the latent state

LABEL_NAMES = {
    IDX_A0: "A0",
    IDX_A1: "A1",
    IDX_A2: "A2",
    IDX_P0: "P0",
    IDX_P1: "P1",
    IDX_P2: "P2",
    IDX_N: "N",
    IDX_DELTA: "Delta",
    IDX_PHI: "Phi",
    IDX_RHO: "Rho",
    IDX_LAMBDA: "Lambda",
    IDX_ETA: "Eta",
    IDX_BETA_1: "Beta 1",
    IDX_BETA_2: "Beta 2",
    IDX_BETA_3: "Beta 3",
    IDX_BETA_4: "Beta_4",
    IDX_X: "Latent State",
}


def plot_means_over_time(model_data, filter_data, filter_stds=None, ci_level=0.95):
    """
    Plots the precomputed mean values over time for:
    - Attractions
    - Probabilities
    - Experience
    - Parameters

    Assumes model_data and filter_data are arrays of shape (T, D),
    where D = total number of dimensions (11).
    """

    model_data = np.array(model_data)
    filter_data = np.array(filter_data)
    time = np.arange(model_data.shape[0])
    D = model_data.shape[1]

    ncols = 4
    nrows = int(np.ceil(D / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axes = axes.flatten()

    for idx in range(D):
        ax = axes[idx]
        name = LABEL_NAMES.get(idx, f"Idx{idx}")
        ax.plot(time, model_data[:, idx], label="Model")
        ax.plot(time, filter_data[:, idx], linestyle="--", label="Filter")

        if filter_stds is not None:
            z = 1.96  # for 95%
            lower = filter_data[:, idx] - z * filter_stds[:, idx]
            upper = filter_data[:, idx] + z * filter_stds[:, idx]
            ax.fill_between(time, lower, upper, color="gray", alpha=0.3, label="95% CI")

        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Value")
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for j in range(D, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Model vs Filter Means Over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("filter_means.png")
    plt.show()


def plot_differences_over_time(model_data, filter_data):
    """
    Plots the difference (model - filter) for each variable over time.

    Assumes model_data and filter_data are arrays of shape (T, D),
    where D = total number of tracked dimensions (11).
    """

    import matplotlib.pyplot as plt
    import numpy as np

    model_data = np.array(model_data)
    filter_data = np.array(filter_data)
    difference = model_data - filter_data  # shape (T, D)

    time = np.arange(model_data.shape[0])
    D = model_data.shape[1]

    ncols = 4
    nrows = int(np.ceil(D / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axes = axes.flatten()

    for idx in range(D):
        ax = axes[idx]
        name = LABEL_NAMES.get(idx, f"Idx{idx}")
        ax.plot(time, difference[:, idx], label=f"Model - Filter")
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"Difference: {name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Difference")
        ax.grid(True)
        ax.legend()

    # Hide any unused subplots
    for j in range(D, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Model vs Filter Differences Over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_latent_and_probs(results, save_prefix="filter_results"):
    """
    Plot latent state (X) and action probabilities (P0, P1, P2) over time,
    comparing filter estimates to the true latent state and opponent probabilities.

    Produces a combined 4-row figure with a single master legend.
    """

    latent_states = np.array(results["latent_states"]).reshape(-1)
    opponent_probs = np.array(results["opponent_prob_record"])
    filter_means = np.array(results["filter_means"])
    filter_p5 = np.array(results["filter_p5"])
    filter_p95 = np.array(results["filter_p95"])

    T = len(latent_states)
    time = np.arange(T)

    # Align lengths by taking the *last* T entries
    if len(filter_means) > T:
        filter_means = filter_means[-T:]
        filter_p5 = filter_p5[-T:]
        filter_p95 = filter_p95[-T:]

    if len(opponent_probs) > T:
        opponent_probs = opponent_probs[-T:]

    # Explicit mapping (requires global constants IDX_X, IDX_P0, IDX_P1, IDX_P2)
    plots = [
        (
            "Distance Between Satellites",
            latent_states,
            filter_means[:, IDX_X],
            filter_p5[:, IDX_X],
            filter_p95[:, IDX_X],
            "True Value",
        ),
        (
            "Probability of Attacker Intercepting",
            opponent_probs[:, 0],
            filter_means[:, IDX_P0],
            filter_p5[:, IDX_P0],
            filter_p95[:, IDX_P0],
            "Opponent Prob",
        ),
        (
            "Probability of Attacker Holding Position",
            opponent_probs[:, 1],
            filter_means[:, IDX_P1],
            filter_p5[:, IDX_P1],
            filter_p95[:, IDX_P1],
            "Opponent Prob",
        ),
        (
            "Probability of Attacker Retreating",
            opponent_probs[:, 2],
            filter_means[:, IDX_P2],
            filter_p5[:, IDX_P2],
            filter_p95[:, IDX_P2],
            "Opponent Prob",
        ),
    ]

    # --- Combined figure ---
    fig, axes = plt.subplots(4, 1, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    handles, labels = None, None
    for ax, (name, true_vals, filt_mean, filt_p5, filt_p95, true_label) in zip(
        axes, plots
    ):
        if true_label is None:
            (h1,) = ax.plot(time, true_vals, label="True Value")
        else:
            (h1,) = ax.plot(time, true_vals, label=true_label)

        (h2,) = ax.plot(time, filt_mean, linestyle="--", label="Filter Mean")
        ax.fill_between(
            time, filt_p5, filt_p95, color="gray", alpha=0.3, label="5–95% CI"
        )

        ax.set_title(name)
        ax.set_ylabel("Value")
        ax.grid(True)

        # Collect handles/labels only once (avoids duplicates)
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()

    # Add one master legend for the whole figure
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
    )

    axes[-1].set_xlabel("Time")
    # plt.suptitle(
    #     f"Posterior Evolution: Latent State and Action Probabilities",
    #     fontsize=16,
    #     y=1.08,
    # )

    combined_path = f"filter_results_4x1.png"
    combined_path_pdf = f"filter_results_4x1.pdf"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.savefig(combined_path_pdf, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined figure: {combined_path}_4x1")

    # --- Combined figure ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    handles, labels = None, None
    for ax, (name, true_vals, filt_mean, filt_p5, filt_p95, true_label) in zip(
        axes, plots
    ):
        if true_label is None:
            (h1,) = ax.plot(time, true_vals, label="True Value")
        else:
            (h1,) = ax.plot(time, true_vals, label=true_label)

        (h2,) = ax.plot(time, filt_mean, linestyle="--", label="Filter Mean")
        ax.fill_between(
            time, filt_p5, filt_p95, color="gray", alpha=0.3, label="5–95% CI"
        )

        ax.set_title(name)
        ax.set_ylabel("Value")
        ax.grid(True)

        # Collect handles/labels only once (avoids duplicates)
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()

    # Add one master legend for the whole figure
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
    )

    axes[-1].set_xlabel("Time")
    plt.suptitle(
        f"Posterior Evolution: Latent State and Action Probabilities",
        fontsize=16,
        y=1.08,
    )

    combined_path = f"filter_results_2x2.png"
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved combined figure: {combined_path}_2x2")

    # --- Individual plots ---
    for name, true_vals, filt_mean, filt_p5, filt_p95, true_label in plots:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, true_vals, label=true_label)
        ax.plot(time, filt_mean, linestyle="--", label="Filter Mean")
        ax.fill_between(
            time, filt_p5, filt_p95, color="gray", alpha=0.3, label="5–95% CI"
        )
        # ax.set_title(f"{results['policy']} – {name}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True)
        if name == "Distance Between Satellites":
            ax.legend()
        path_single = f"{save_prefix}_{name.replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(path_single, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Saved individual figure: {path_single}")


def plot_all_latent_state_trajectories(
    results_all,
    summaries,
    replicate=0,
    rename_policies=None,
    style_map=None,
    savepath="combined_trajectories.png",
):
    """
    Plot latent state trajectories for all policies with customizable labels and styles.

    Args:
        results_all (list[dict]): List of results dictionaries (one per policy)
        summaries (list[dict]): List of summary dictionaries (unused, but kept for compatibility)
        replicate (int): Replicate index for labeling
        rename_policies (dict): Mapping {raw_name: pretty_name}
        style_map (dict): Mapping {policy: {"color": ..., "linestyle": ..., "linewidth": ...}}
        savepath (str): Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(12, 6))

    for results, summary in zip(results_all, summaries):
        latent_states = np.array(results["latent_states"])
        policy = results["policy"]

        # Get pretty label if provided
        label = rename_policies.get(policy, policy) if rename_policies else policy

        # Get style if provided
        style = style_map.get(policy, {}) if style_map else {}
        color = style.get("color", None)
        linestyle = style.get("linestyle", "-")
        linewidth = style.get("linewidth", 2)

        plt.plot(
            latent_states,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.9,
            zorder=3 if policy == "MC" else 1,
        )

    # Use the first policy's betas just for bound visualization
    atk_betas = results_all[0]["attacker_betas"]
    def_betas = results_all[0]["defender_betas"]
    atk_bounds = (atk_betas[0], atk_betas[1])
    def_bounds = (def_betas[0], def_betas[1])

    plt.axhspan(atk_bounds[0], atk_bounds[1], color="red", alpha=0.1)
    plt.axhspan(def_bounds[0], def_bounds[1], color="blue", alpha=0.1)

    plt.axhline(atk_bounds[0], color="red", linestyle="--", linewidth=2)
    plt.axhline(atk_bounds[1], color="red", linestyle="--", linewidth=2)
    plt.axhline(def_bounds[0], color="blue", linestyle="--", linewidth=2)
    plt.axhline(def_bounds[1], color="blue", linestyle="--", linewidth=2)

    # plt.text(len(latent_states)-1, atk_bounds[1], "Attacker MAD", color="red", va="bottom", ha="right")
    # plt.text(len(latent_states)-1, atk_bounds[0], "Attacker MSD", color="red", va="top", ha="right")

    # plt.text(len(latent_states)-1, def_bounds[1], "Defender MAD", color="blue", va="bottom", ha="left")
    # plt.text(len(latent_states)-1, def_bounds[0], "Defender MSD", color="blue", va="top", ha="left")

    # --- Offsets for text placement (as a fraction of the range)
    y_offset = 0.02 * (
        max(def_betas[1], atk_betas[1]) - min(def_betas[0], atk_betas[0])
    )

    # Attacker labels (shifted inside the band)
    plt.text(
        len(latent_states) * 0.99,
        atk_bounds[1] - y_offset,
        "Attacker MAD",
        color="red",
        va="top",
        ha="right",
    )
    plt.text(
        len(latent_states) * 0.99,
        atk_bounds[0] + y_offset,
        "Attacker MSD",
        color="red",
        va="bottom",
        ha="right",
    )

    # Defender labels (shifted inside the band)
    plt.text(
        len(latent_states) * 1.01,  # tiny push outside for symmetry
        def_bounds[1] - y_offset,
        "Defender MAD",
        color="blue",
        va="top",
        ha="right",
    )
    plt.text(
        len(latent_states) * 1.01,
        def_bounds[0] + y_offset,
        "Defender MSD",
        color="blue",
        va="bottom",
        ha="right",
    )

    plt.xlabel("Timestep")
    plt.ylabel("Distance between satellites")
    plt.legend(
    loc="upper center",       # position relative to bbox_to_anchor
    bbox_to_anchor=(0.5, -0.1),  # center below the axes
    ncol=6                    # number of columns (horizontal)
    )
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_policy_action_heatmaps(
    results_all,
    savepath="defender_actions_heatmap.png",
    rename_policies=None,
    action_labels=None,
):
    """
    Plot heatmap of defender actions over time across policies.

    Args:
        results_all (list[dict]): List of results dictionaries (one per policy)
        savepath (str): Where to save the output PNG
        rename_policies (dict): Optional mapping {old_name: new_name}
        action_labels (dict): Optional mapping {action_number: label}
    """

    # Collect defender actions
    policy_names = [res["policy"] for res in results_all]
    defender_actions = [np.array(res["actions"])[:, 1] for res in results_all]
    defender_matrix = np.vstack(defender_actions)

    # Setup discrete colormap
    num_actions = int(defender_matrix.max()) + 1
    cmap = ListedColormap(sns.color_palette("Set2", num_actions))
    norm = BoundaryNorm(np.arange(-0.5, num_actions + 0.5), cmap.N)

    fig, ax = plt.subplots(figsize=(12, 6))
    hm = sns.heatmap(
        defender_matrix,
        cmap=cmap,
        norm=norm,
        cbar=True,
        cbar_kws={"ticks": np.arange(num_actions)},
        xticklabels=10,
        yticklabels=policy_names,
        linewidths=0.01,
        linecolor="gray",
        ax=ax,
    )

    # Rename y-axis labels if desired
    if rename_policies:
        ax.set_yticklabels([rename_policies.get(name, name) for name in policy_names])

    # Update colorbar tick labels
    cbar = hm.collections[0].colorbar
    if action_labels:
        labels = [action_labels.get(i, str(i)) for i in range(num_actions)]
        cbar.set_ticklabels(labels)
    else:
        cbar.set_ticklabels([str(i) for i in range(num_actions)])

    # ax.set_title("Defender Actions Over Time", fontsize=14, pad=12)
    ax.set_ylabel("Policy", fontsize=12)
    ax.set_xlabel("Timestep", fontsize=12)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    with open("single_evaluation_output_results.pkl", "rb") as f:
        results_all = pickle.load(f)

    with open("single_evaluation_output_summaries.pkl", "rb") as f:
        summaries_all = pickle.load(f)

    res = results_all[0]  # e.g. Boltzmann

    # Suppose you want to rename policies
    custom_names = {
        "Boltzmann_Reduced_K": "Boltzmann",
        "H2S (H=1)": "H2S",
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
        results_all,
        summaries_all,
        rename_policies=custom_names,
        style_map=line_style_map,
    )
    plot_policy_action_heatmaps(
        results_all, rename_policies=custom_names, action_labels=action_labels
    )
