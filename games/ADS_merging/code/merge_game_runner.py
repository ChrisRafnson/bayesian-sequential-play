import time
import numpy as np
import jax
import jax.numpy as jnp

import filters as jf
import models
from config import MainConfig
from game_managers import ADSGameManager
from policies import get_means
from ADP import get_ADP_action_MLP
from utility_functions import utility
from constants import (
    LOGICAL_PARTICLE_DIM,
    IDX_XD1, IDX_XD2, IDX_XD3, IDX_XD4,
    IDX_XA1, IDX_XA2, IDX_XA3, IDX_XA4,
)



def run_merge_game_episode(
    *,
    config: MainConfig,
    model_seq,
    policy: str,
    seed: int,
):
    """
    Run ONE merge-game episode on a fixed config and policy.
    Returns a dict of rich episode data.
    """

    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    manager = ADSGameManager(config)
    ewa_model = models.MannedVehicle(config)

    # Initialize particles
    particles = jf.initialize_particles_beta_prior(
        subkey,
        config,
        config.filter.num_particles,
        config.game.num_attacker_actions,
    )

    num_steps = config.game.num_timesteps
    defender_actions = jnp.arange(config.game.num_defender_actions)

    # Logs
    step_logs = []
    traj_def, traj_att = [], []

    for t in range(num_steps):
        mean_particle = get_means(particles)

        key, subkey = jax.random.split(key)

        if policy == "ADP":
            defender_action = get_ADP_action_MLP(
                mean_particle,
                model_seq[t],
                defender_actions,
                config,
            )
        else:
            raise ValueError(f"Unsupported policy {policy}")

        attacker_action = ewa_model.select_action() 

        # Step environment
        defender_obs, attacker_obs = manager.step(
            attacker_action,
            defender_action,
        )

        # Update particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            defender_obs,
            config,
        )

        # Unpack true state
        XD1, XD2, XD3, XD4, XA1, XA2, XA3, XA4 = manager.state

        # Store trajectories
        traj_def.append([XD1, XD2])
        traj_att.append([XA1, XA2])

        # Defender utility
        def_util = utility(
            XD2,
            XD1, XD2, XD3, XD4,
            XA1, XA2, XA3, XA4,
            defender_action,
            config.game.defender_beta_1,
            config.game.defender_beta_2,
            config.game.defender_beta_3,
            config.game.defender_beta_4,
            config.game.defender_beta_5,
            config.game.defender_beta_6,
            config,
            is_defender=True,
        )

        # Attacker utility
        atk_util = utility(
            XD2,
            XA1, XA2, XA3, XA4,
            XD1, XD2, XD3, XD4,
            defender_action,
            config.model.beta_A1,
            config.model.beta_A2,
            config.model.beta_A3,
            config.model.beta_A4,
            config.model.beta_A5,
            config.model.beta_A6,
            config,
            is_defender=False,
        )

        # Filter statistics
        filter_means = {
            # Defender belief means
            "filter_def_x": float(jnp.mean(particles[:, IDX_XD1])),
            "filter_def_y": float(jnp.mean(particles[:, IDX_XD2])),
            "filter_def_heading": float(jnp.mean(particles[:, IDX_XD3])),
            "filter_def_speed": float(jnp.mean(particles[:, IDX_XD4])),

            # Attacker belief means
            "filter_att_x": float(jnp.mean(particles[:, IDX_XA1])),
            "filter_att_y": float(jnp.mean(particles[:, IDX_XA2])),
            "filter_att_heading": float(jnp.mean(particles[:, IDX_XA3])),
            "filter_att_speed": float(jnp.mean(particles[:, IDX_XA4])),
        }


        step_logs.append({
            "t": t,

            # Actions
            "defender_action": int(defender_action),
            "attacker_action": int(attacker_action),

            # Utility
            "defender_utility": float(def_util),
            "attacker_utility": float(atk_util),

            # True defender state
            "def_x": float(XD1),
            "def_y": float(XD2),
            "def_heading": float(XD3),
            "def_speed": float(XD4),

            # True attacker state
            "att_x": float(XA1),
            "att_y": float(XA2),
            "att_heading": float(XA3),
            "att_speed": float(XA4),

            # Filter belief means
            **filter_means,
        })


    return {
        "policy": policy,
        "seed": seed,
        "step_logs": step_logs,
        "traj_def": np.asarray(traj_def),
        "traj_att": np.asarray(traj_att),
    }
