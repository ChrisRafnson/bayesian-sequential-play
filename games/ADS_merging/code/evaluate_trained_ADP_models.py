import json
import pickle
from pathlib import Path
import jax
from tqdm import tqdm

from regressions import load_state_sequence
from merge_game_runner import run_merge_game_episode
from constants import LOGICAL_PARTICLE_DIM


MODELS_DIR = Path("/home/chris/Desktop/bayesian-security-games/games/ADS_merging/code/adp_architecture_sweep_test")
INSTANCES_DIR = Path("/home/chris/Desktop/bayesian-security-games/games/ADS_merging/code/experiment_instances")
OUT_DIR = Path("/home/chris/Desktop/bayesian-security-games/games/ADS_merging/code/experiment_results_EWA_test")
OUT_DIR.mkdir(exist_ok=True)

import numpy as np
import jax.numpy as jnp

def to_json_safe(obj):
    """
    Recursively convert numpy / jax objects to JSON-serializable types.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]

    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]

    elif isinstance(obj, (np.ndarray, jnp.ndarray)):
        return obj.tolist()

    elif isinstance(obj, (np.float32, np.float64, jnp.float32, jnp.float64)):
        return float(obj)

    elif isinstance(obj, (np.int32, np.int64, jnp.int32, jnp.int64)):
        return int(obj)

    else:
        return obj


def load_instance(path):
    with open(path, "rb") as f:
        return pickle.load(f)



def evaluate_all_models():

    model_dirs = sorted(
        [d for d in MODELS_DIR.iterdir() if d.is_dir()]
    )
    instance_files = sorted(INSTANCES_DIR.glob("i*.pkl"))

    if len(model_dirs) == 0:
        raise RuntimeError("No trained models found.")

    if len(instance_files) == 0:
        raise RuntimeError("No test instances found.")

    # Outer progress bar: models
    for model_dir in tqdm(
        model_dirs,
        desc="Evaluating models",
        unit="model",
    ):
        state_path = model_dir / "state_sequence.msgpack"
        meta_path = model_dir / "meta.json"

        if not state_path.exists() or not meta_path.exists():
            tqdm.write(f"[SKIP] Missing files in {model_dir.name}")
            continue

        meta = json.loads(meta_path.read_text())

        # Load one instance just to get num_timesteps
        cfg_example = load_instance(instance_files[0])

        # Load trained ADP model ONCE per architecture
        model_seq = load_state_sequence(
            filepath=state_path,
            num_stages=cfg_example.game.num_timesteps,
            input_dim=LOGICAL_PARTICLE_DIM + 2,
            hidden_dims=meta["hidden_dims"],
            rng=jax.random.PRNGKey(0),
        )

        # Inner progress bar: instances
        for inst_id, inst_path in enumerate(
            tqdm(
                instance_files,
                desc=f"Instances for {model_dir.name}",
                unit="instance",
                leave=False,
            )
        ):
            out_file = OUT_DIR / f"{model_dir.name}__{inst_path.stem}.json"

            # Resume / skip logic
            if out_file.exists():
                continue

            cfg = load_instance(inst_path)

            # Deterministic evaluation seed
            seed = meta["seed"] + inst_id

            try:
                results = run_merge_game_episode(
                    config=cfg,
                    model_seq=model_seq,
                    policy="ADP",
                    seed=seed,
                )

                out = {
                    "model": model_dir.name,
                    "instance": inst_path.name,
                    "meta": meta,
                    "results": results,
                }

                out_safe = to_json_safe(out)
                out_file.write_text(json.dumps(out_safe, indent=2))

            except Exception as e:
                tqdm.write(
                    f"[ERROR] {model_dir.name} | {inst_path.name}: {e}"
                )
                continue

def evaluate_single_model(model_dir: Path):
    assert model_dir.exists() and model_dir.is_dir()

    instance_files = sorted(INSTANCES_DIR.glob("i*.pkl"))
    if len(instance_files) == 0:
        raise RuntimeError("No test instances found.")

    state_path = model_dir / "state_sequence.msgpack"
    meta_path  = model_dir / "meta.json"

    if not state_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Missing files in {model_dir.name}")

    meta = json.loads(meta_path.read_text())
    cfg_example = load_instance(instance_files[0])

    model_seq = load_state_sequence(
        filepath=state_path,
        num_stages=cfg_example.game.num_timesteps,
        input_dim=LOGICAL_PARTICLE_DIM + 2,
        hidden_dims=meta["hidden_dims"],
        rng=jax.random.PRNGKey(0),
    )

    for inst_id, inst_path in enumerate(
        tqdm(instance_files, desc=f"Instances for {model_dir.name}")
    ):
        out_file = OUT_DIR / f"{model_dir.name}__{inst_path.stem}.json"
        # if out_file.exists():
        #     continue

        cfg = load_instance(inst_path)
        seed = meta["seed"] + inst_id

        results = run_merge_game_episode(
            config=cfg,
            model_seq=model_seq,
            policy="ADP",
            seed=seed,
        )

        out = {
            "model": model_dir.name,
            "instance": inst_path.name,
            "meta": meta,
            "results": results,
        }

        out_file.write_text(
            json.dumps(to_json_safe(out), indent=2)
        )



if __name__ == "__main__":
    # evaluate_all_models()
    evaluate_single_model(
    MODELS_DIR / "d2_w32_actrelu_lr1e-03_seed52120"
    )

