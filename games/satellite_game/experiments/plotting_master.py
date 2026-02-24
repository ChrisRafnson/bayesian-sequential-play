#!/usr/bin/env python3
"""
Evaluation Explorer
-------------------
One-stop EDA script for full_evaluation_results.csv.

Usage:
    python evaluation_explorer.py --csv /path/to/full_evaluation_results.csv --out ./plots --remove-fp

Notes:
- Saves PNGs into --out
- Parses config_file -> opponent_policy, beta_level, instance_number
- Safely parses list-like columns for time-series and action distributions
"""

import argparse
import ast
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set(style="whitegrid")


# ---------------------------
# Helpers
# ---------------------------
def safe_eval(val):
    """Safely convert a string that looks like a list/tuple/dict into Python object."""
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return val
    return val


def parse_config(config_str):
    """
    Parse config_file assuming a pattern like 'OppPolicy_beta_<lvl>_inst_<n>'.
    Adjust if your pattern differs.
    """
    if not isinstance(config_str, str):
        return pd.Series(
            {"opponent_policy": None, "beta_level": None, "instance_number": None}
        )
    parts = config_str.split("_")
    opp = parts[0] if len(parts) > 0 else None
    beta, inst = None, None
    ints = [int(p) for p in parts if p.isdigit()]
    if len(ints) >= 1:
        beta = ints[0]
    if len(ints) >= 2:
        inst = ints[1]
    return pd.Series(
        {"opponent_policy": opp, "beta_level": beta, "instance_number": inst}
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def ci95(series: pd.Series):
    """Compute 95% CI half-width assuming normality (for large n)."""
    x = series.dropna().values
    n = len(x)
    if n == 0:
        return np.nan
    m = x.mean()
    s = x.std(ddof=1) if n > 1 else 0.0
    hw = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, hw


# ---------------------------
# Plotting Routines
# ---------------------------
def plot_util_box_and_kde(df, outdir):
    # Boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="policy", y="def_avg_util", showfliers=False)
    plt.title("Defender Average Utility by Policy")
    plt.xlabel("Policy")
    plt.ylabel("def_avg_util")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_def_util_box.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="policy", y="atk_avg_util", showfliers=False)
    plt.title("Attacker Average Utility by Policy")
    plt.xlabel("Policy")
    plt.ylabel("atk_avg_util")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02_atk_util_box.png"))
    plt.close()

    # KDEs
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="def_avg_util", hue="policy", fill=True, common_norm=False)
    plt.title("Defender Utility Distributions by Policy")
    plt.xlabel("def_avg_util")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02c_def_util_kde.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="atk_avg_util", hue="policy", fill=True, common_norm=False)
    plt.title("Attacker Utility Distributions by Policy")
    plt.xlabel("atk_avg_util")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "02d_atk_util_kde.png"))
    plt.close()


def plot_opponent_beta_effects(df, outdir):
    g = sns.catplot(
        data=df,
        x="policy",
        y="def_avg_util",
        hue="beta_level",
        col="opponent_policy",
        kind="box",
        col_wrap=3,
        height=4,
        aspect=1.3,
        showfliers=False,
    )
    g.set_titles("Opponent: {col_name}")
    g.set_axis_labels("Policy", "def_avg_util")
    plt.tight_layout()
    g.figure.savefig(os.path.join(outdir, "04_def_util_by_opp_beta.png"))
    plt.close(g.figure)

    # Heatmap of mean defender util policy x opponent
    pivot = df.groupby(["policy", "opponent_policy"], as_index=False)[
        "def_avg_util"
    ].mean()
    heat = pivot.pivot(index="policy", columns="opponent_policy", values="def_avg_util")
    plt.figure(figsize=(1.5 * len(heat.columns) + 3, 0.5 * len(heat.index) + 3))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="vlag", center=0)
    plt.title("Mean Defender Utility: Policy x Opponent")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "04b_def_util_heatmap_policy_x_opp.png"))
    plt.close()


def plot_time_dynamics(df, outdir):
    df2 = df.copy()
    df2["def_win_means"] = df2["def_win_means"].apply(safe_eval)
    df2["atk_win_means"] = df2["atk_win_means"].apply(safe_eval)

    def_avg = df2.groupby("policy")["def_win_means"].apply(
        lambda s: np.mean(
            np.stack(
                [np.array(x) for x in s if isinstance(x, (list, tuple, np.ndarray))]
            ),
            axis=0,
        )
    )
    atk_avg = df2.groupby("policy")["atk_win_means"].apply(
        lambda s: np.mean(
            np.stack(
                [np.array(x) for x in s if isinstance(x, (list, tuple, np.ndarray))]
            ),
            axis=0,
        )
    )

    plt.figure(figsize=(12, 6))
    cmap = plt.get_cmap("tab10").colors
    for i, pol in enumerate(def_avg.index):
        plt.plot(def_avg[pol], label=f"{pol}", lw=2, color=cmap[i % len(cmap)])
    plt.title("Defender Utility Over Windows (Averaged by Policy)")
    plt.xlabel("Window Index")
    plt.ylabel("def_win_means")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_def_time_dynamics.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    for i, pol in enumerate(atk_avg.index):
        plt.plot(atk_avg[pol], label=f"{pol}", lw=2, ls="--", color=cmap[i % len(cmap)])
    plt.title("Attacker Utility Over Windows (Averaged by Policy)")
    plt.xlabel("Window Index")
    plt.ylabel("atk_win_means")
    plt.legend(ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "06_atk_time_dynamics.png"))
    plt.close()


def plot_action_distributions(df, outdir):
    df2 = df.copy()
    df2["atk_action_dist"] = df2["atk_action_dist"].apply(safe_eval)
    df2["def_action_dist"] = df2["def_action_dist"].apply(safe_eval)

    atk_avg = df2.groupby("policy")["atk_action_dist"].apply(
        lambda s: np.mean(
            np.stack(
                [np.array(x) for x in s if isinstance(x, (list, tuple, np.ndarray))]
            ),
            axis=0,
        )
    )
    def_avg = df2.groupby("policy")["def_action_dist"].apply(
        lambda s: np.mean(
            np.stack(
                [np.array(x) for x in s if isinstance(x, (list, tuple, np.ndarray))]
            ),
            axis=0,
        )
    )

    for name, series in [("attacker", atk_avg), ("defender", def_avg)]:
        rows = []
        for pol, vec in zip(series.index, series.values):
            if isinstance(vec, (list, tuple, np.ndarray)):
                for a_idx, p in enumerate(vec):
                    rows.append({"policy": pol, "action": a_idx, "prob": p})
        long = pd.DataFrame(rows)
        if long.empty:
            continue
        plt.figure(figsize=(10, 6))
        sns.barplot(data=long, x="action", y="prob", hue="policy")
        plt.title(f"Average {name.capitalize()} Action Distribution by Policy")
        plt.xlabel("Action")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"07_{name}_action_dist.png"))
        plt.close()


def plot_bounds_distributions(df, outdir):
    plt.figure(figsize=(10, 5))
    sns.kdeplot(
        data=df, x="time_in_def_bounds", hue="policy", fill=True, common_norm=False
    )
    plt.title("Time in Defender Bounds (Distribution)")
    plt.xlabel("Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "08_time_in_def_bounds_kde.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.kdeplot(
        data=df, x="time_in_atk_bounds", hue="policy", fill=True, common_norm=False
    )
    plt.title("Time in Attacker Bounds (Distribution)")
    plt.xlabel("Proportion")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "08_time_in_atk_bounds_kde.png"))
    plt.close()


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="full_evaluation_results.csv",
        help="Path to CSV file",
    )
    parser.add_argument(
        "--out", type=str, default="./plots", help="Output directory for PNGs"
    )
    parser.add_argument(
        "--remove-fp", action="store_true", help="Exclude FP policy from all plots"
    )
    args = parser.parse_args()

    ensure_dir(args.out)
    df = pd.read_csv(args.csv)

    numeric_cols = [
        "avg_comp_time",
        "atk_avg_util",
        "atk_tot_util",
        "def_avg_util",
        "def_tot_util",
        "time_in_def_bounds",
        "time_in_atk_bounds",
        "window_size",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "config_file" in df.columns:
        parsed = df["config_file"].apply(parse_config)
        df = pd.concat([df, parsed], axis=1)

    if args.remove_fp and "policy" in df.columns:
        df = df[df["policy"] != "FP"].copy()

    core_cols = [
        c
        for c in ["policy", "def_avg_util", "atk_avg_util", "avg_comp_time"]
        if c in df.columns
    ]
    df = df.dropna(subset=core_cols)

    plot_util_box_and_kde(df, args.out)
    if {"opponent_policy", "beta_level"}.issubset(df.columns):
        plot_opponent_beta_effects(df, args.out)
    plot_time_dynamics(df, args.out)
    plot_action_distributions(df, args.out)
    plot_bounds_distributions(df, args.out)

    print(f"Done. Plots saved to: {args.out}")


if __name__ == "__main__":
    main()
