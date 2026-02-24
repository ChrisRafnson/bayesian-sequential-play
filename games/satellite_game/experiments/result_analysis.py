import ast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler

df = pd.read_csv(
    "/home/chris/Desktop/bayesian-security-games/full_evaluation_results.csv"
)


def parse_config(config_str):
    parts = config_str.split("_")
    return pd.Series(
        {
            "opponent_policy": parts[0],
            "beta_level": int(parts[3]),
            "instance_number": int(parts[5]),
        }
    )


# Apply to your DataFrame (assuming the column is called 'config')
parsed_config = df["config_file"].apply(parse_config)

# Merge new columns into your original DataFrame
df = pd.concat([df, parsed_config], axis=1)

# Convert relevant columns to numeric
df["atk_avg_util"] = pd.to_numeric(df["atk_avg_util"], errors="coerce")
df["def_avg_util"] = pd.to_numeric(df["def_avg_util"], errors="coerce")
df["avg_comp_time"] = pd.to_numeric(df["avg_comp_time"], errors="coerce")

scaler = RobustScaler()
df["def_avg_util_scaled"] = scaler.fit_transform(df[["def_avg_util"]])

Q1 = df["def_avg_util"].quantile(0.25)
Q3 = df["def_avg_util"].quantile(0.75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

df["def_avg_util_clipped"] = df["def_avg_util"].clip(lower, upper)

# Set up a general style
sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x="policy", y="def_avg_util_clipped")
plt.title("Defender Average Utility by Policy")
plt.ylabel("Average Utility")

plt.tight_layout()
plt.savefig("plot1.png")
plt.close()


# Convert string to 3x3 matrix
def parse_joint_matrix(row):
    try:
        return np.array(ast.literal_eval(row.strip()))
    except:
        return np.full((3, 3), np.nan)


# Plot average joint action distribution by policy
policies = df["policy"].unique()
for policy in policies:
    subset = df[df["policy"] == policy]
    joint_matrices = np.stack(
        subset["joint_action_dist"].dropna().apply(parse_joint_matrix)
    )
    avg_joint_matrix = np.nanmean(joint_matrices, axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(avg_joint_matrix, annot=True, cmap="Blues", fmt=".2f", cbar=True)
    plt.title(f"Avg. Joint Action Distribution\n({policy})")
    plt.xlabel("Defender Actions")
    plt.ylabel("Attacker Actions")
    plt.savefig(f"plot_dist_{policy}.png")


# Safe eval in case columns are already parsed lists
def safe_eval(val):
    if isinstance(val, str):
        return ast.literal_eval(val)
    return val


# Clean column names and apply safe parsing
df.columns = df.columns.str.strip()
df["atk_win_means"] = df["atk_win_means"].apply(safe_eval)
df["def_win_means"] = df["def_win_means"].apply(safe_eval)

# Compute averages
atk_time_avg = df.groupby("policy")["atk_win_means"].apply(
    lambda x: np.mean(np.stack(x), axis=0)
)
def_time_avg = df.groupby("policy")["def_win_means"].apply(
    lambda x: np.mean(np.stack(x), axis=0)
)

# Colorblind-friendly palette (Tableau 10)
colors = plt.get_cmap("tab10").colors
color_map = {
    policy: colors[i % len(colors)] for i, policy in enumerate(atk_time_avg.index)
}

# Plot 1: Attacker utility
plt.figure(figsize=(10, 5))
for i, policy in enumerate(atk_time_avg.index):
    plt.plot(
        atk_time_avg[policy], label=f"{policy} - Attacker", color=color_map[policy]
    )
plt.title("Attacker Average Utility Over Time")
plt.xlabel("Window Index")
plt.ylabel("Utility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot2.png")


# Plot 2: Defender utility
plt.figure(figsize=(10, 5))
for i, policy in enumerate(def_time_avg.index):
    plt.plot(
        def_time_avg[policy],
        label=f"{policy} - Defender",
        linestyle="--",
        color=color_map[policy],
    )
plt.title("Defender Average Utility Over Time")
plt.xlabel("Window Index")
plt.ylabel("Utility")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot3.png")

plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="policy", y="time_in_def_bounds")
plt.title("Time in Defender Bounds by Policy")
plt.ylabel("Timesteps in Bounds")
plt.tight_layout()
plt.savefig("plot4.png")
