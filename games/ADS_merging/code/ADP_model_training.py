import itertools
import json
import os
import jax
from jax import random
from tqdm import tqdm
from tqdm import trange

from ADP import compute_value_functions
from config_ADP_prior import MainConfig
from regressions import save_params_sequence
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# -----------------------------
# Experiment grid
# -----------------------------
# DEPTHS = [2, 4]
# WIDTHS = [32, 128, 256]
# ACTIVATIONS = ["relu", "tanh", "sigmoid"]
# LEARNING_RATES = [1e-2, 1e-3, 1e-4]

DEPTHS = [2]
WIDTHS = [32]
ACTIVATIONS = ["relu"]
LEARNING_RATES = [1e-3]

BASE_DIR = "/home/chris/Desktop/bayesian-security-games/games/ADS_merging/code/adp_architecture_sweep_test"
os.makedirs(BASE_DIR, exist_ok=True)


# -----------------------------
# Deterministic factorial seed
# -----------------------------
def factorial_seed(depth, width, activation, lr, replicate=0):
    act_map = {"relu": 1, "tanh": 2, "sigmoid": 3}
    lr_map = {1e-2: 1, 1e-3: 2, 1e-4: 3}

    return int(
        10_000 * depth +
        1_000 * width +
        100 * act_map[activation] +
        10 * lr_map[lr] +
        replicate
    )

# -----------------------------
# Helper: check completion
# -----------------------------
def run_complete(exp_dir):
    return (
        os.path.exists(os.path.join(exp_dir, "state_sequence.msgpack")) and
        os.path.exists(os.path.join(exp_dir, "meta.json"))
    )

# -----------------------------
# Main loop
# -----------------------------
grid = list(itertools.product(
    DEPTHS, WIDTHS, ACTIVATIONS, LEARNING_RATES
))

for depth, width, act, lr in tqdm(
    grid,
    desc="ADP architecture sweep",
    unit="model"
):
    hidden_dims = [width] * depth
    seed = factorial_seed(depth, width, act, lr)

    exp_name = f"d{depth}_w{width}_act{act}_lr{lr:.0e}_seed{seed}"
    exp_dir = os.path.join(BASE_DIR, exp_name)

    # if run_complete(exp_dir):
    #     tqdm.write(f"[SKIP] {exp_name} already completed")
    #     continue

    tqdm.write(f"=== Training {exp_name} ===")
    os.makedirs(exp_dir, exist_ok=True)

    # Deterministic setup
    key = random.PRNGKey(seed)
    key, cfg_key = random.split(key)

    config = MainConfig.create(cfg_key)

    try:
        model_seq = compute_value_functions(
            key,
            config,
            num_timesteps=config.game.num_timesteps,
            hidden_dims=hidden_dims,
            activation=act,
            lr=lr,
            num_epochs=1000,
        )

        # Save sequence
        save_params_sequence(
            model_seq,
            os.path.join(exp_dir, "state_sequence.msgpack"),
        )

        # Save metadata
        meta = {
            "depth": depth,
            "width": width,
            "hidden_dims": hidden_dims,
            "activation": act,
            "learning_rate": lr,
            "seed": seed,
            "num_timesteps": config.game.num_timesteps,
        }

        with open(os.path.join(exp_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[DONE] {exp_name}")

    except Exception as e:
        print(f"[ERROR] {exp_name}: {e}")
        # Leave directory intact so you can inspect logs or rerun
        continue
