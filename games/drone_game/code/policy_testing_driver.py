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
    evaluate_DM_policy,
    generate_all_trajectories,
)
from tqdm import tqdm
import regressions as rf

# ------------------- Paths --------------------
CONFIG_DIR = Path(
    "/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_game_instances_TEST"
)
MODEL_DIR = Path(
    "/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_ADP_models"
)
OUTPUT_CSV = Path("drone_results_oops.csv")

# ------------------- Settings --------------------
NUM_REPLICATES = 1
BASE_SEED = 42

# ------------------- Policies --------------------
POLICIES = {
    "Boltzmann":  lambda key, seed, config, _: evaluate_boltzmann_policy(key, seed, config),
    # "H2S-H1": lambda key, seed, config, _: evaluate_H2S_policy(
    #     key, seed, config, H=1
    # ),
    "H2S-H3":     lambda key, seed, config, _: evaluate_H2S_policy(key, seed, config, H=3),
    "AMG": lambda key, seed, config, _: evaluate_AMG_policy(key, seed, config),
    "ADP": lambda key, seed, config, model_path: evaluate_ADP_policy(key, seed, config, model_path),
    "MC": lambda key, seed, config, _: evaluate_MC_policy(key, seed, config),
    # "FP": lambda key, seed, config, _: evaluate_FP_policy(key, seed, config),
    # "DM": lambda key, seed, config, _: evaluate_DM_policy(key, seed, config, tao=1),
}

# ------------------- Helper Functions --------------------

def get_model_path_for_config(config_name: str) -> Path:
    """
    Given a config filename stem like 'SEED_1_TIGHT(2,3,1)i0',
    return the expected corresponding ADP model path.
    """
    return MODEL_DIR / f"{config_name}ADP_MODEL.msgpack"


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
        "window_size",
        "window_ids",
        "atk_win_means",
        "def_win_means",
        "adv_win_means",
        "attacker_actions",
        "defender_actions",
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
        tqdm.write(f"Evaluating config: {config_name}")

        with config_path.open("rb") as f:
            config = pickle.load(f)

        for policy_name, evaluator in tqdm(
            POLICIES.items(), desc="Policies", leave=False
        ):
            tqdm.write(f"Evaluating policy: {policy_name}")

            # Determine model path *just once per config/policy*
            if policy_name == "ADP":
                model_path = get_model_path_for_config(config_name)
                if not model_path.exists():
                    tqdm.write(f"⚠️ Model file missing for {config_name}: {model_path}")
                    continue
            else:
                model_path = None  # Not needed for non-ADP policies

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
                    result = evaluator(rep_key, np_seed, config, model_path)
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
                    tqdm.write(f"     ↪ FAILED: {e}")

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

def test_file_reads():
    config_paths = sorted(CONFIG_DIR.glob("*.pkl"))
    base_key = jax.random.PRNGKey(BASE_SEED)

    # Begin evaluations
    for config_path in config_paths:
        config_name = config_path.stem
        print(f"Reading config: {config_name}")


# ------------------- Entry --------------------

if __name__ == "__main__":
    # test_file_reads()
    run_evaluations()
    # test_key_gen()
