import csv
import hashlib
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from config import MainConfig
from evaluation import (
    compute_episode_summary,
    evaluate_ADP_policy,
    evaluate_AMG_policy,
    evaluate_boltzmann_policy,
    evaluate_FP_policy,
    evaluate_H2S_policy,
    evaluate_MC_policy,
    evaluate_reduced_k_boltzmann_policy,
    generate_all_trajectories,
)
from tqdm import tqdm

# ------------------- Paths --------------------
CONFIG_DIR = Path(
    "/home/chris/Desktop/bayesian-security-games/Code/Rafnson/satellite_game_instances_new"
)
MODEL_MATRIX_PATH = Path(
    "/home/chris/Desktop/bayesian-security-games/model_matrix_OLS.npy"
)
OUTPUT_CSV = Path("full_evaluation_results_new_only_filter_policies.csv")

# ------------------- Settings --------------------
NUM_REPLICATES = 5
BASE_SEED = 42

# ------------------- Trajectories --------------------
H1_trajectories = generate_all_trajectories(H=1)
# H2_trajectories = generate_all_trajectories(H=2)

# ------------------- Policies --------------------
POLICIES = {
    # "Boltzmann":  lambda key, seed, config, _: evaluate_boltzmann_policy(key, seed, config),
    "Boltzmann_reduced_k": lambda key, seed, config, _: evaluate_reduced_k_boltzmann_policy(
        key, seed, config
    ),
    "H2S-H1": lambda key, seed, config, _: evaluate_H2S_policy(
        key, seed, config, trajectories=H1_trajectories, H=1
    ),
    # "H2S-H2":     lambda key, seed, config, _: evaluate_H2S_policy(key, seed, config, trajectories=H2_trajectories, H=2),
    "AMG": lambda key, seed, config, _: evaluate_AMG_policy(key, seed, config),
    "ADP": lambda key, seed, config, mm: evaluate_ADP_policy(key, seed, config, mm),
    "MC": lambda key, seed, config, _: evaluate_MC_policy(key, seed, config),
    "FP": lambda key, seed, config, _: evaluate_FP_policy(key, seed, config),
}

# ------------------- Helper Functions --------------------


def stable_hash(x: str) -> int:
    # Turn any input into a stable 32-bit integer
    h = hashlib.sha256(str(x).encode("utf-8")).hexdigest()
    return int(h, 16) & 0xFFFFFFFF


def fold_in_key(base_key, *args):
    for arg in args:
        base_key = jax.random.fold_in(base_key, stable_hash(arg) & 0xFFFFFFFF)
    return base_key


def load_completed_rows(path: Path):
    if not path.exists():
        return set()
    with path.open("r") as f:
        reader = csv.DictReader(f)
        return {
            (row["config_file"], row["policy"], int(row["replicate"])) for row in reader
        }


# ------------------- Main Evaluation Loop --------------------


def run_evaluations():
    total_start_time = time.perf_counter()
    config_paths = sorted(CONFIG_DIR.glob("*.pkl"))
    model_matrix = jnp.array(np.load(MODEL_MATRIX_PATH))
    base_key = jax.random.PRNGKey(BASE_SEED)

    # Load prior results if resuming
    completed_rows = load_completed_rows(OUTPUT_CSV)

    # Prepare CSV
    fieldnames = [
        "config_file",
        "policy",
        "replicate",
        "seed",
        "avg_comp_time",
        "atk_avg_util",
        "atk_tot_util",
        "def_avg_util",
        "def_tot_util",
        "atk_action_dist",
        "def_action_dist",
        "joint_action_dist",
        "time_in_def_bounds",
        "time_in_atk_bounds",
        "window_size",
        "window_ids",
        "atk_win_means",
        "def_win_means",
        "adv_win_means",
        "attacker_actions",
        "defender_actions",
        "latent_state_record",
        "attacker_betas",
        "defender_betas",
        "attacker_utilities",
        "defender_utilities",
    ]

    file_exists = OUTPUT_CSV.exists()
    csvfile = OUTPUT_CSV.open("a", newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    if not file_exists:
        writer.writeheader()

    # Begin evaluations
    for config_path in tqdm(config_paths, desc="Configs"):
        config_name = config_path.stem
        print(f"Evaluating config: {config_name}")

        with config_path.open("rb") as f:
            config = pickle.load(f)

        for policy_name, evaluator in tqdm(
            POLICIES.items(), desc="Policies", leave=False
        ):

            for r in tqdm(range(NUM_REPLICATES), desc="Replicates", leave=False):
                if (config_name, policy_name, r) in completed_rows:
                    print(f"Skipping {config_name}_{policy_name}_{r}...")
                    continue
                rep_start = time.perf_counter()

                # Create a deterministic subkey per (config, policy, replicate)
                rep_key = fold_in_key(base_key, config_name, policy_name, r)

                try:
                    # Convert Unique JAX key to NumPy seed
                    maxval = jnp.array(2**32 - 1, dtype=jnp.uint32)  # max uint32 value
                    np_seed = int(
                        jax.random.randint(
                            rep_key, (), minval=0, maxval=maxval, dtype=jnp.uint32
                        )
                    )
                    result = evaluator(rep_key, np_seed, config, model_matrix)
                    summary = compute_episode_summary(result, config)
                    summary.update(
                        {
                            "config_file": config_name,
                            "policy": policy_name,
                            "replicate": r,
                            "seed": np_seed,
                        }
                    )
                    writer.writerow(summary)
                    csvfile.flush()
                except Exception as e:
                    print(f"     ↪ FAILED: {e}")

                rep_end = time.perf_counter()

    csvfile.close()
    total_end_time = time.perf_counter()
    print(f"\n✅ All evaluations complete. Saved to: {OUTPUT_CSV}")
    print(f"⏱️ Total time: {total_end_time - total_start_time:.2f} sec")


def test_key_gen():
    config_paths = sorted(CONFIG_DIR.glob("*.pkl"))
    base_key = jax.random.PRNGKey(BASE_SEED)

    # Begin evaluations
    for config_path in config_paths:
        config_name = config_path.stem
        print(f"Evaluating config: {config_name}")

        with config_path.open("rb") as f:
            config = pickle.load(f)

        for policy_name, evaluator in POLICIES.items():
            print(f" → Policy: {policy_name}")

            for r in range(NUM_REPLICATES):

                # Create a deterministic subkey per (config, policy, replicate)
                rep_key = fold_in_key(base_key, config_name, policy_name, r)
                maxval = jnp.array(2**32 - 1, dtype=jnp.uint32)  # max uint32 value
                np_seed = int(
                    jax.random.randint(
                        rep_key, (), minval=0, maxval=maxval, dtype=jnp.uint32
                    )
                )

                print(f"Replicate Key: {rep_key}, Numpy seed: {np_seed}")


# ------------------- Entry --------------------

if __name__ == "__main__":
    run_evaluations()
    # test_key_gen()
