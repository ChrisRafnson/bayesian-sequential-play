from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path
import jax
import jax.numpy as jnp
import jax.random as random
from config import MainConfig
from ADP import compute_value_functions
import regressions as rf


def stable_hash(x: str) -> int:
    # Turn any input into a stable 32-bit integer
    h = hashlib.sha256(str(x).encode("utf-8")).hexdigest()
    return int(h, 16) & 0xFFFFFFFF

def instance_key(base_key, instance: int):
    """
    Unique, reproducible PRNGKey for (learner, beta_level, instance).
    """
    k = random.fold_in(base_key, instance)
    return k

def generate_instances(
    base_seed: int = 42,
    num_instances_per_level: int = 5,
    out_dir: str | Path = "./merge_game_instances",
    skip_if_exists: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_key = random.PRNGKey(base_seed)
    manifest = []

    for instance in range(num_instances_per_level):

        # Filename
        name = (
            f"i{instance}.pkl"
        )
        fpath = out_dir / name

        # Stable, per-item key
        subkey = instance_key(base_key, instance)

        # Create a fresh config with payoff bounds applied
        cfg = MainConfig.create(subkey)


        if skip_if_exists and fpath.exists():
            # Still record it in the manifest for completeness
            manifest.append(
                {
                    "file": str(fpath),
                    "instance": instance,
                    "base_seed": base_seed,
                }
            )
            continue

        with fpath.open("wb") as f:
            pickle.dump(cfg, f)

        manifest.append(
            {
                "file": str(fpath),
                "instance": instance,
                "base_seed": base_seed,
            }
        )

        # print_main_config(cfg)

    # Write a simple manifest for reproducibility and auditing
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} configs to {out_dir.resolve()}")
    return manifest

if __name__ == "__main__":
    generate_instances(
        base_seed=42,
        num_instances_per_level=25,
        out_dir="/home/chris/Desktop/bayesian-security-games/games/ADS_merging/code/experiment_instances_EWA_test",
        skip_if_exists=False,
    )

