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
from config import print_main_config
from ADP import compute_value_functions
import regressions as rf

#Experiment Levels?
# NUM_ATK_ACTIONS = (1, 2)
NUM_ATK_ACTIONS = [1]
DEF_PRIOR_TYPES = ("FLAT", "TIGHT")
PRIOR_MARGIN = 0.1 #The tightness of the prior when it is not flat
EPSILON_LEVELS = [0.0]
FIXED_MODEL_KEYS = [0]
A_LEVELS = [1.0, 5.0, 7.0]
B_LEVELS = [1.0, 5.0, 7.0]
C_LEVELS = [1.0, 5.0, 7.0]

NUM_SITES = 8

#Side-note, the initial attraction value is equal to the nummber of entering edges to a vertex in the graph
FIXED_ATTRACTIONS = jnp.array([
    2.0, #Site 1
    1.0, #Site 2
    2.0, #Site 3
    1.0, #Site 4
    2.0, #Site 5
    2.0, #Site 6
    4.0, #Site 7
    1.0  #Site 8
])

# ---- Learner parameterizations (EWA variants) -------------------------------
# Names chosen to match your labels: CCRL / ACRL / WBL / RED / SPEC
LEARNERS = (
    # "CCRL",
    # "ACRL",
    # "WBL",
    # "RED"
    "FIXED", # A specific EWA learner drawn outside the instance generation
    )



def apply_overrides_for_learner(learner: str, cfg, fixed_model=None):
    """
    Returns a new config with learner-specific (delta, rho, experience) overrides.
    Uses the freshly-sampled 'phi' from cfg.model when needed.
    """
    model = cfg.model

    if learner == "FIXED" and fixed_model is not None:
        #Replaces the drawn EWA model with the constant EWA model drawn earlier
        return replace(cfg, model=fixed_model)

    elif learner == "CCRL":
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
    
    elif learner == "SPEC":
        delta = jnp.array(0.0, dtype=jnp.float32)
        phi = jnp.array(0.0, dtype=jnp.float32)
        rho = jnp.array(0.0, dtype=jnp.float32)
        exp0 = jnp.array(1.0, dtype=jnp.float32)
        Lambda = jnp.array(0.0, dtype=jnp.float32)
        model = replace(model, delta=delta, rho=rho, experience=exp0)
        return replace(cfg, model=model)

    elif learner == "RED":
        # Random EWA Draw: leave whatever MainConfig.create sampled
        return cfg

    else:
        raise ValueError(f"Unknown learner: {learner}")

def stable_hash(x: str) -> int:
    # Turn any input into a stable 32-bit integer
    h = hashlib.sha256(str(x).encode("utf-8")).hexdigest()
    return int(h, 16) & 0xFFFFFFFF

def instance_key(base_key, learner: str, num_atk_actions: int, prior_type: str, epsilon: float, a :float, b:float, c:float , instance: int):
    """
    Unique, reproducible PRNGKey for (learner, beta_level, instance).
    """
    k = random.fold_in(base_key, stable_hash(learner) & 0xFFFFFFFF)
    k = random.fold_in(base_key, stable_hash(prior_type) & 0xFFFFFFFF)
    k = random.fold_in(k, num_atk_actions)
    k = random.fold_in(k, epsilon)
    k = random.fold_in(k, a)
    k = random.fold_in(k, b)
    k = random.fold_in(k, c)
    k = random.fold_in(k, instance)
    return k

def generate_instances(
    base_seed: int = 42,
    num_instances_per_level: int = 5,
    out_dir: str | Path = "./drone_game_instances",
    skip_if_exists: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_key = random.PRNGKey(base_seed)
    manifest = []

    for key in FIXED_MODEL_KEYS:

        #Draw specific learner
        fixed_key = jax.random.PRNGKey(key)
        fixed_model = MainConfig.create(fixed_key, game_kwargs={"num_sites": NUM_SITES}, model_kwargs={"initial_attractions":FIXED_ATTRACTIONS}).model

        for learner in LEARNERS:
            for num_atk_actions in NUM_ATK_ACTIONS:
                for prior_type in DEF_PRIOR_TYPES:
                    for epsilon in EPSILON_LEVELS:
                        for a in A_LEVELS:
                            for b in B_LEVELS:
                                for c in C_LEVELS:
                                    for instance in range(num_instances_per_level):

                                        # Filename
                                        name = (
                                            f"SEED_{key}_"
                                            f"{prior_type}"
                                            f"{a,b,c}"
                                            f"i{instance}.pkl"
                                        )
                                        fpath = out_dir / name

                                        # Stable, per-item key
                                        subkey = instance_key(base_key, learner, num_atk_actions, prior_type, epsilon, a, b, c, instance)

                                        # Create a fresh config with payoff bounds applied
                                        cfg = MainConfig.create(
                                            subkey,
                                            game_kwargs={
                                                "num_sites": NUM_SITES,
                                                "num_attacker_selections": num_atk_actions,
                                                "utility_a": a,
                                                "utility_b": b,
                                                "utility_c": c},
                                            model_kwargs={"epsilon": jnp.array(epsilon)}
                                        )

                                        # Apply learner-specific overrides (delta, rho, experience)
                                        cfg = apply_overrides_for_learner(learner, cfg, fixed_model)

                                        # If defender prior is TIGHT → shrink defender beta bounds ±0.5 around model’s true betas
                                        if prior_type == "TIGHT":
                                            # Extract model parameter values
                                            model_vals = {
                                                "delta": float(cfg.model.delta),
                                                "phi": float(cfg.model.phi),
                                                "rho": float(cfg.model.rho),
                                                "Lambda": float(cfg.model.Lambda),
                                                "experience": float(cfg.model.experience),
                                                "beta": float(cfg.model.beta),
                                            }

                                            # Define a tightening margin
                                            margin = PRIOR_MARGIN

                                            # Compute tight bounds around each parameter
                                            tight_bounds = {}
                                            for name, val in model_vals.items():
                                                lower = max(0, val - margin)
                                                upper = val + margin
                                                tight_bounds[f"{name}_bounds"] = (lower, upper)

                                            # Replace the priors with these tightened bounds
                                            cfg = replace(
                                                cfg,
                                                priors=replace(
                                                    cfg.priors,
                                                    delta_bounds=tight_bounds["delta_bounds"],
                                                    phi_bounds=tight_bounds["phi_bounds"],
                                                    rho_bounds=tight_bounds["rho_bounds"],
                                                    lambda_bounds=tight_bounds["Lambda_bounds"],
                                                    experience_bounds=tight_bounds["experience_bounds"],
                                                    beta_bounds=tight_bounds["beta_bounds"],
                                                ),
                                            )

                                        if skip_if_exists and fpath.exists():
                                            # Still record it in the manifest for completeness
                                            manifest.append(
                                                {
                                                    "file": str(fpath),
                                                    "learner": learner,
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
                                                "instance": instance,
                                                "base_seed": base_seed,
                                            }
                                        )

                                        # print_main_config(cfg)

    # Write a simple manifest for reproducibility and auditing
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} configs to {out_dir.resolve()}")
    return manifest

def generate_ADP_models(    
    base_seed: int = 42,
    out_dir: str | Path = "./drone_game_ADP_models",
    num_epochs=100
    ):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = random.PRNGKey(base_seed)
    manifest = []

    for num_atk_actions in NUM_ATK_ACTIONS:

        # Filename
        name = (
            f"A{num_atk_actions}_ADP_model_V11.msgpack"
        )
        fpath = out_dir / name


        key, subkey = jax.random.split(key)
        cfg = MainConfig.create(
            subkey,
            game_kwargs={"num_sites": NUM_SITES, "num_attacker_selections": num_atk_actions}
        )

        # Generate models
        print("Training ADP Models...")
        key, subkey = jax.random.split(key)
        model_seq = compute_value_functions(subkey, cfg, cfg.game.num_timesteps, num_epochs=num_epochs)

        # Save models
        rf.save_params_sequence([m.params for m in model_seq], fpath)


def generate_instances_train_models(
    base_seed: int = 42,
    num_instances_per_level: int = 5,
    out_dir: str | Path = "./drone_game_instances",
    skip_if_exists: bool = False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ADP_dir = Path("/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_ADP_models")
    ADP_dir.mkdir(parents=True, exist_ok=True)

    ADP_key = random.PRNGKey(base_seed + 69)

    base_key = random.PRNGKey(base_seed)
    manifest = []

    for key in FIXED_MODEL_KEYS:

        #Draw specific learner
        fixed_key = jax.random.PRNGKey(key)
        fixed_model = MainConfig.create(fixed_key, game_kwargs={"num_sites": NUM_SITES}, model_kwargs={"initial_attractions":FIXED_ATTRACTIONS}).model

        for learner in LEARNERS:
            for num_atk_actions in NUM_ATK_ACTIONS:
                for prior_type in DEF_PRIOR_TYPES:
                    for epsilon in EPSILON_LEVELS:
                        for a in A_LEVELS:
                            for b in B_LEVELS:
                                for c in C_LEVELS:
                                    for instance in range(num_instances_per_level):

                                        # Filename
                                        name = (
                                            f"SEED_{key}_"
                                            f"{prior_type}"
                                            f"{a,b,c}"
                                            f"i{instance}.pkl"
                                        )
                                        fpath = out_dir / name

                                        # Stable, per-item key
                                        subkey = instance_key(base_key, learner, num_atk_actions, prior_type, epsilon, a, b, c, instance)

                                        # Create a fresh config with payoff bounds applied
                                        cfg = MainConfig.create(
                                            subkey,
                                            game_kwargs={
                                                "num_sites": NUM_SITES,
                                                "num_attacker_selections": num_atk_actions,
                                                "utility_a": a,
                                                "utility_b": b,
                                                "utility_c": c},
                                            model_kwargs={"epsilon": jnp.array(epsilon)}
                                        )

                                        # Apply learner-specific overrides (delta, rho, experience)
                                        cfg = apply_overrides_for_learner(learner, cfg, fixed_model)

                                        # If defender prior is TIGHT → shrink defender beta bounds ±0.5 around model’s true betas
                                        if prior_type == "TIGHT":
                                            # Extract model parameter values
                                            model_vals = {
                                                "delta": float(cfg.model.delta),
                                                "phi": float(cfg.model.phi),
                                                "rho": float(cfg.model.rho),
                                                "Lambda": float(cfg.model.Lambda),
                                                "experience": float(cfg.model.experience),
                                                "beta": float(cfg.model.beta),
                                            }

                                            # Define a tightening margin
                                            margin = PRIOR_MARGIN

                                            # Compute tight bounds around each parameter
                                            tight_bounds = {}
                                            for name, val in model_vals.items():
                                                lower = max(0, val - margin)
                                                upper = val + margin
                                                tight_bounds[f"{name}_bounds"] = (lower, upper)

                                            # Replace the priors with these tightened bounds
                                            cfg = replace(
                                                cfg,
                                                priors=replace(
                                                    cfg.priors,
                                                    delta_bounds=tight_bounds["delta_bounds"],
                                                    phi_bounds=tight_bounds["phi_bounds"],
                                                    rho_bounds=tight_bounds["rho_bounds"],
                                                    lambda_bounds=tight_bounds["Lambda_bounds"],
                                                    experience_bounds=tight_bounds["experience_bounds"],
                                                    beta_bounds=tight_bounds["beta_bounds"],
                                                ),
                                            )

                                        if skip_if_exists and fpath.exists():
                                            # Still record it in the manifest for completeness
                                            manifest.append(
                                                {
                                                    "file": str(fpath),
                                                    "learner": learner,
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
                                                "instance": instance,
                                                "base_seed": base_seed,
                                            }
                                        )

                                        """
                                        Now we can go ahead and train an ADP model for this configuration, the model is trained on
                                        the first instance of each factor level combination, and then reused for the other instances
                                        """

                                        if instance == 0:
                                            # Filename
                                            name = (
                                                f"SEED_{key}_"
                                                f"{prior_type}"
                                                f"{a,b,c}"
                                                f"i{instance}"
                                                f"ADP_MODEL.msgpack"
                                            )
                                            fpath = ADP_dir / name

                                            # Generate models
                                            print("Training ADP Models...")
                                            ADP_key, subkey = jax.random.split(ADP_key)
                                            model_seq = compute_value_functions(subkey, cfg, cfg.game.num_timesteps, num_epochs=500)

                                            # Save models
                                            rf.save_params_sequence([m.params for m in model_seq], fpath)
                                        else:

                                            #Since it is another instance of a level, we can reuse the first instances ADP model
                                            # Filename
                                            name = (
                                                f"SEED_{key}_"
                                                f"{prior_type}"
                                                f"{a,b,c}"
                                                f"i{instance}"
                                                f"ADP_MODEL.msgpack"
                                            )
                                            fpath = ADP_dir / name

                                            # Save models
                                            rf.save_params_sequence([m.params for m in model_seq], fpath)


                                        # print_main_config(cfg)

    # Write a simple manifest for reproducibility and auditing
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest)} configs to {out_dir.resolve()}")
    return manifest



if __name__ == "__main__":
    # generate_instances(
    #     base_seed=42,
    #     num_instances_per_level=10,
    #     out_dir="/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_game_instances_TEST",
    #     skip_if_exists=False,
    # )

    # generate_ADP_models(
    #     base_seed=42,
    #     out_dir="/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_ADP_models",
    #     num_epochs=500
    # )

    generate_instances_train_models(
        base_seed=42,
        num_instances_per_level=10,
        out_dir="/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_game_instances_TEST",
        skip_if_exists=False,
    )
