import os
import ast
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="whitegrid")

output_dir = "/home/chris/Desktop/bayesian-security-games/games/drone_game/plots"
os.makedirs(output_dir, exist_ok=True)


# --- Parse config_file ---
def parse_config_name(name: str):
    """
    Example: 'ACRL_A1_FLAT_eps0.00_i0'
    → attacker_type='ACRL', n_allowed_attacks=1, prior='FLAT', epsilon=0.00, instance=0
    """
    if pd.isna(name):
        return {"attacker_type": None, "n_allowed_attacks": None, "prior": None, "epsilon": None, "instance": None}

    m = re.match(r"SEED_(\d+)_([A-Za-z]+)_A(\d+)_([A-Za-z]+)_eps([0-9.]+)_i(\d+)", name)
    if m:
        return {
            "model_seed":m.group(1),
            "attacker_type": m.group(2),
            "n_allowed_attacks": int(m.group(3)),
            "prior": m.group(4),
            "epsilon": float(m.group(5)),
            "instance": int(m.group(6)),
        }
    # fallback: partial parse if format differs slightly
    tokens = name.split("_")
    return {
        "attacker_type": tokens[0] if len(tokens) > 0 else None,
        "n_allowed_attacks": int(tokens[1][1:]) if len(tokens) > 1 and tokens[1].startswith("A") else None,
        "prior": tokens[2] if len(tokens) > 2 else None,
        "epsilon": float(tokens[3].replace("eps", "")) if len(tokens) > 3 and "eps" in tokens[3] else None,
        "instance": int(tokens[4].replace("i", "")) if len(tokens) > 4 and tokens[4].startswith("i") else None,
    }

def safe_parse_array(x):
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x))
        except (ValueError, SyntaxError):
            return np.nan  # or leave it as-is if you prefer
    elif isinstance(x, (list, np.ndarray)):
        return np.array(x)
    else:
        return np.nan
    
# If your column still looks like "[0.0, 0.0, -2.0, ...]"
def parse_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x

for i in range(1, 12):

    result_version = i

    # Path to your CSV file
    csv_path = f"/home/chris/Desktop/bayesian-security-games/drone_results_V{result_version}.csv"

    df = pd.read_csv(csv_path)

    parsed = df["config_file"].apply(parse_config_name)
    parsed_df = pd.DataFrame(parsed.tolist())

    # --- Merge parsed fields ---
    df = pd.concat([df, parsed_df], axis=1)

    df = df.dropna(axis=1, how="all")

    # Ensure epsilon and prior are treated as categorical for plotting
    df["epsilon"] = df["epsilon"].astype(str)
    df["prior"] = df["prior"].astype(str)

    # Separate data by distince EWA draws
    df["model_seed"] = df["model_seed"].astype(str)

    df = df[df["model_seed"] == "0"].copy()

    #Split dataframes into games with single and two action choices
    df["model_seed"] = df["model_seed"].astype(str)

    # Create a FacetGrid: columns by epsilon, rows by prior
    g = sns.catplot(
        data=df,
        x="policy",
        y="def_tot_util",
        col="epsilon",
        row="prior",
        kind="box",
        sharey=True,
        showfliers=False,
        height=4,
        aspect=1.2
    )

    g.set_axis_labels("Policy", "Average Defender Utility")
    g.set_titles(row_template="Prior: {row_name}", col_template="Epsilon: {col_name}")
    plt.tight_layout()

    g.savefig(f"{output_dir}/V{result_version}_boxplot.png")  # <-- SAVE HERE
    # plt.show()

    df["def_action_dist"] = df["def_action_dist"].apply(safe_parse_array)
    df["atk_action_dist"] = df["atk_action_dist"].apply(safe_parse_array)

    # Compute mean distributions across replicates per policy
    atk_means = df.groupby("policy")["atk_action_dist"].apply(lambda x: np.mean(np.stack(x.values), axis=0))
    def_means = df.groupby("policy")["def_action_dist"].apply(lambda x: np.mean(np.stack(x.values), axis=0))

    # Plot defender heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(np.vstack(def_means.values), cmap="Blues", annot=False,
                xticklabels=[f"Action {i}" for i in range(len(def_means.iloc[0]))],
                yticklabels=def_means.index)

    plt.title("Defender Action Distribution by Policy")
    plt.xlabel("Action Index")
    plt.ylabel("Policy")
    plt.tight_layout()

    plt.savefig(f"{output_dir}/V{result_version}_heatmap.png")  # <-- SAVE
    # plt.show()

    df["defender_utilities"] = df["defender_utilities"].apply(parse_list)

    print(type(df["defender_utilities"].iloc[0]))


    df_long = df.explode("defender_utilities").reset_index(drop=True)

    df_long["timestep"] = df_long.groupby(["config_file","policy", "replicate"]).cumcount()

    df_long.rename(columns={"defender_utilities": "def_util"}, inplace=True)

    # Plot non-cumulative (utility over time)
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df_long,
        x="timestep",
        y="def_util",
        hue="policy",
    )
    plt.title("Defender Utility Over Time by Policy")
    plt.xlabel("Timestep")
    plt.ylabel("Defender Utility")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/V{result_version}_utility_over_time.png")
    plt.close()   # <-- IMPORTANT: clears the figure

    # Now compute cumulative utility
    df_long["def_util"] = df_long["def_util"].astype(float)
    df_long["cum_def_util"] = (
        df_long.groupby(["policy", "config_file", "replicate"])["def_util"].cumsum()
    )

    # Plot cumulative utility (fresh figure)
    plt.figure(figsize=(10,6))
    sns.lineplot(
        data=df_long,
        x="timestep",
        y="cum_def_util",
        hue="policy",
    )
    plt.title("Cumulative Defender Utility Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Utility")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/V{result_version}_cumulative_utility.png")
    plt.close()