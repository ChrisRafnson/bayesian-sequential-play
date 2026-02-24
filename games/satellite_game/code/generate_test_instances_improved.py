from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
from config import MainConfig

# Payoff parameter bounds per "beta level"
BETA_LEVELS = [
    {
        "beta_MSD": (10.0, 30.0),
        "beta_MAD": (40.0, 60.0),
        "beta_C": (10.0, 30.0),
        "beta_penalty": (0.0, 1.0),
    },  # Level 0 (Co-op overlap)
    {
        "beta_MSD": (40.0, 60.0),
        "beta_MAD": (70.0, 90.0),
        "beta_C": (10.0, 30.0),
        "beta_penalty": (0.0, 1.0),
    },  # Level 1 (Adversarial disjoint)
    {
        "beta_MSD": (5.0, 20.0),
        "beta_MAD": (50.0, 70.0),
        "beta_C": (10.0, 30.0),
        "beta_penalty": (0.0, 1.0),
    },  # Level 2 (Contains defender)
    {
        "beta_MSD": (20.0, 30.0),
        "beta_MAD": (40.0, 50.0),
        "beta_C": (10.0, 30.0),
        "beta_penalty": (0.0, 1.0),
    },  # Level 3 (Contained by defender)
]

# ---- Learner parameterizations (EWA variants) -------------------------------
# Names chosen to match your labels: CCRL / ACRL / WBL / RED
LEARNERS = ("CCRL", "ACRL", "WBL", "RED")


def apply_overrides_for_learner(learner: str, cfg):
    """
    Returns a new config with learner-specific (delta, rho, experience) overrides.
    Uses the freshly-sampled 'phi' from cfg.model when needed.
    """
    model = cfg.model

    if learner == "CCRL":
        # Cumulative Choice RL: delta=0, rho=0, N(0)=1
        delta = jnp.array(0.0, dtype=jnp.float32)
        rho = jnp.array(0.0, dtype=jnp.float32)
        exp0 = jnp.array(1.0, dtype=jnp.float32)
        model = replace(model, delta=delta, rho=rho, experience=exp0)
        return replace(cfg, model=model)

    elif learner == "ACRL":
        # Average Choice RL: rho = phi, N(0) = 1 / (1 - rho), delta=0
        delta = jnp.array(0.0, dtype=jnp.float32)
        rho = jnp.asarray(model.phi, dtype=jnp.float32)
        exp0 = jnp.asarray(1.0 / (1.0 - rho), dtype=jnp.float32)
        model = replace(model, delta=delta, rho=rho, experience=exp0)
        return replace(cfg, model=model)

    elif learner == "WBL":
        # Weighted Belief Learning: delta=1, rho=phi
        delta = jnp.array(1.0, dtype=jnp.float32)
        rho = jnp.asarray(model.phi, dtype=jnp.float32)
        model = replace(model, delta=delta, rho=rho)
        return replace(cfg, model=model)

    elif learner == "RED":
        # Random EWA Draw: leave whatever MainConfig.create sampled
        return cfg

    else:
        raise ValueError(f"Unknown learner: {learner}")


def apply_defender_beta_override(betas, cfg):
    """
    Returns a new config with defender-specific beta overrides.
    """
    game = cfg.game

    game = replace(
        game,
        defender_beta_MSD=betas[0],
        defender_beta_MAD=betas[1],
        defender_beta_C=betas[2],
        defender_beta_penalty=betas[3],
    )

    return replace(cfg, game=game)


def stable_hash(x: str) -> int:
    # Turn any input into a stable 32-bit integer
    h = hashlib.sha256(str(x).encode("utf-8")).hexdigest()
    return int(h, 16) & 0xFFFFFFFF


def instance_key(base_key, learner: str, beta_level: int, instance: int):
    """
    Unique, reproducible PRNGKey for (learner, beta_level, instance).
    """
    k = random.fold_in(base_key, stable_hash(learner) & 0xFFFFFFFF)
    k = random.fold_in(k, beta_level)
    k = random.fold_in(k, instance)
    return k


def generate_instances(
    base_seed: int = 42,
    num_instances_per_level: int = 5,
    out_dir: str | Path = "./satellite_game_instances",
    skip_if_exists: bool = False,
    defender_betas=[20.0, 50.0, 100.0, 0.1],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_key = random.PRNGKey(base_seed)
    manifest = []

    for learner in LEARNERS:
        for beta_level, bounds in enumerate(BETA_LEVELS):
            for instance in range(num_instances_per_level):

                # Stable, per-item key
                subkey = instance_key(base_key, learner, beta_level, instance)

                # Create a fresh config with payoff bounds applied
                cfg = MainConfig.create(subkey, model_kwargs={"model_bounds": bounds})

                # Apply learner-specific overrides (delta, rho, experience)
                cfg = apply_overrides_for_learner(learner, cfg)

                cfg = apply_defender_beta_override(defender_betas, cfg)

                # Filename
                name = f"{learner}_beta_level_{beta_level}_instance_{instance}.pkl"
                fpath = out_dir / name

                if skip_if_exists and fpath.exists():
                    # Still record it in the manifest for completeness
                    manifest.append(
                        {
                            "file": str(fpath),
                            "learner": learner,
                            "beta_level": beta_level,
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
                        "learner": learner,
                        "beta_level": beta_level,
                        "instance": instance,
                        "base_seed": base_seed,
                    }
                )

    # Write a simple manifest for reproducibility and auditing
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} configs to {out_dir.resolve()}")
    return manifest


if __name__ == "__main__":
    generate_instances(
        base_seed=42,
        num_instances_per_level=100,
        out_dir="/home/chris/Desktop/bayesian-security-games/games/satellite_game/satellite_game_instances_new",
        skip_if_exists=False,
    )
