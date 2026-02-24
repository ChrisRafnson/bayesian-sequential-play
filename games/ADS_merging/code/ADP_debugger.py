import time
import pandas as pd
import filters as jf
import jax
import jax.numpy as jnp
import models
import numpy as np
import matplotlib.pyplot as plt
import plotting as plotting
from config_ADP_prior import MainConfig
from benchmark_policies import get_MC_action_jax
from policies import (
    generate_all_trajectories,
    get_means,
)
from utility_functions import utility, utility_debugger
from constants import (
    LOGICAL_PARTICLE_DIM
)
from ADP import (
    compute_value_functions,
    get_ADP_action_MLP,
    adp_compute_data,
    get_value_estimate_with_model,
    _one_hot_def_actions, 
    generate_cost_function_estimate)
import regressions as rf
from game_managers import ADSGameManager
from constants import HIDDEN_DIMS
from filters import print_particle_with_labels
import os

def print_action_decomp(particle, t, model_seq, config):
    D = jnp.arange(config.game.num_defender_actions)

    # cost-only (no NN)
    costs = jax.vmap(generate_cost_function_estimate, in_axes=(None, 0, None))(
        particle[:LOGICAL_PARTICLE_DIM], D, config
    )

    # NN-only (continuation): call model_seq[t] directly on features, without adding cost again
    # Build X exactly like training did
    a_cont = config.game.defender_action_set[D]  # (K,2)
    a_norm = jnp.stack(
        [a_cont[:,0] / config.game.max_acceleration,
         a_cont[:,1] / config.game.max_steering],
        axis=1
    )
    X = jnp.concatenate([jnp.repeat(particle[:LOGICAL_PARTICLE_DIM][None,:], D.shape[0], axis=0), a_norm], axis=1)

    state = model_seq[t]  # <-- IMPORTANT
    Xn = (X - state.X_mean) / state.X_std
    v_norm = state.apply_fn(state.params, Xn).squeeze()
    vhat = v_norm * state.y_std.squeeze() + state.y_mean.squeeze()

    Q = costs + vhat

    best_cost = int(jnp.argmax(costs))
    best_q    = int(jnp.argmax(Q))

    print("best by cost:", best_cost, " best by Q:", best_q)
    for i in range(D.shape[0]):
        print(i, "cost", float(costs[i]), "vhat", float(vhat[i]), "Q", float(Q[i]))



def debug_ADP_output(
    particle: jnp.ndarray,
    defender_action: int,
    state_next,
    config: MainConfig,
) -> jnp.ndarray:
    """
    Value estimate using:
        cost(s,d) + V_hat_{t+1}(s,d)
    where V_hat_{t+1} is the learned MLP for the next timestep.
    """
    # # Feature vector: [particle, defender_action]
    # X = jnp.concatenate([jnp.ravel(particle), jnp.ravel(defender_action)])
    # Xn = (X - state_next.X_mean.squeeze()) / state_next.X_std.squeeze()

    num_defender_actions = config.game.num_defender_actions

    # particle = particle[:LOGICAL_PARTICLE_DIM]
    # a_oh = _one_hot_def_actions(defender_action, num_defender_actions)  # (K,)
    # X = jnp.concatenate([jnp.ravel(particle), jnp.ravel(a_oh)])         # (D+K,)

    # --- Build features (using the continuous action features) ---
    particle = particle[:LOGICAL_PARTICLE_DIM]

   # Look up continuous defender action: (2,)
    a_cont = config.game.defender_action_set[defender_action]

    # Normalization constants (define once in config if possible)
    ACCEL_MAX = config.game.max_acceleration    # max |accel|
    STEER_MAX = config.game.max_steering    # max |steering| (degrees)

    a_norm = jnp.stack(
        [
            a_cont[0] / ACCEL_MAX,
            a_cont[1] / STEER_MAX,
        ],
        axis=0
    )


    X = jnp.concatenate([particle, a_norm], axis=0)

    Xn = (X - state_next.X_mean.squeeze()) / state_next.X_std.squeeze()


    y_norm = state_next.apply_fn(state_next.params, Xn[None, :]).squeeze()
    yhat = y_norm * state_next.y_std.squeeze() + state_next.y_mean.squeeze()
    jax.debug.print("is_terminal={}", state_next.is_terminal)

    return yhat.squeeze()

def print_action_values(values, defender_action_set):
    values = np.asarray(values)
    actions = np.asarray(defender_action_set)

    best = values.argmax()

    print("\nDefender Action Values (best marked with ★)")
    print("------------------------------------------")
    print(f"{'idx':>3} | {'accel':>6} | {'steer':>6} | {'value':>12}")
    print("------------------------------------------")

    for i, ((accel, steer), v) in enumerate(zip(actions, values)):
        star = " ★" if i == best else "  "
        print(f"{i:>3} | {accel:>6.2f} | {steer:>6.2f} | {v:>12.6f}{star}")

def save_adp_debug_data(debug_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for t, data in debug_data.items():
        np.savez(
            os.path.join(save_dir, f"adp_data_t{t}.npz"),
            particles=np.asarray(data["particles"]),
            actions=np.asarray(data["actions"]),
            values=np.asarray(data["values"]),
        )

if __name__ == "__main__":

    seed = 1

    np.random.seed(seed)  # Set seed
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    config = MainConfig.create(subkey)

    num_steps = config.game.num_timesteps

    # input_dimensions = LOGICAL_PARTICLE_DIM + config.game.num_defender_actions #One hot encoding
    input_dimensions = LOGICAL_PARTICLE_DIM + 2

    #Checking model output

    path = "/home/chris/Desktop/bayesian-security-games/games/ADS_merging/ADP_models/mlp_sequence_TEST.msgpack"


    # Generate models
    print("Training ADP Models...")
    key, subkey = jax.random.split(key)
    model_seq, debug_data = compute_value_functions(subkey, config, num_steps, num_epochs=100)
    save_adp_debug_data(debug_data, "/home/chris/Desktop/bayesian-security-games/games/ADS_merging/code/adp_debug_data")


    # # Save models
    rf.save_params_sequence(model_seq, path)

    #Load models
    model_seq = rf.load_state_sequence(
        path,
        num_steps,
        input_dim=input_dimensions,
        hidden_dims=HIDDEN_DIMS,
        rng=jax.random.PRNGKey(0),
    )

    print("len(model_seq) =", len(model_seq))
    for i, m in enumerate(model_seq):
        print(i, m.is_terminal)


    #Test Particle

    test_particle = jnp.array([
        1.0, #Experience
        1.0, #Delta
        1.0, #Phi
        1.0, #Rho
        1.0, #Lambda
        1.0, #Atk beta1
        1.0, #Atk beta2
        1.0, #Atk beta3
        1.0, #Atk beta4
        1.0, #Atk beta5
        1.0, #Atk beta6
        1.0, #Atk obs noise
        -10.0, #Def horizontal
        70.0, #Def vertical
        0.0, #Def heading
        27.0, #Def speed
        2.0, #Atk horizontal
        0.0, #Atk vertical
        0.0, #Atk heading
        27.0, #Atk speed
        5.0, #Attraction 1
        5.0, #Attraction 2
        5.0, #Attraction 3
        5.0, #Attraction 4
        50.0, #Attraction 5
        5.0, #Attraction 6
        5.0, #Attraction 7
        5.0, #Attraction 8
        5.0, #Attraction 9
        0.0, #Probability 1
        0.0, #Probability 2
        0.0, #Probability 3
        0.0, #Probability 4
        1.0, #Probability 5
        0.0, #Probability 6
        0.0, #Probability 7
        0.0, #Probability 8
        0.0, #Probability 9
    ])

    model_output = debug_ADP_output(
        test_particle,
        4,
        model_seq[num_steps-1],
        config
    )

    defender_actions = jnp.arange(config.game.num_defender_actions)

    estimates = jax.vmap(
            get_value_estimate_with_model, in_axes=(None, 0, None, None)
        )(test_particle, defender_actions, model_seq[num_steps-2], config)

    cost_estimates = jax.vmap(
            generate_cost_function_estimate, in_axes=(None, 0, None)
        )(test_particle, defender_actions, config)


    print(model_seq[num_steps-1].y_mean)
    print(model_seq[num_steps-1].y_std)
    print_action_values(cost_estimates, config.game.defender_action_set)
    print_action_values(estimates, config.game.defender_action_set)


    # estimates = jax.vmap(
    #         debug_ADP_output, in_axes=(None, 0, None, None)
    #     )(test_particle, defender_actions, model_seq[num_steps-2], config)

    # cost_estimates = jax.vmap(
    #         generate_cost_function_estimate, in_axes=(None, 0, None)
    #     )(test_particle, defender_actions, config)

    # print(model_seq[num_steps-2].y_mean)
    # print(model_seq[num_steps-2].y_std)
    # print_action_values(estimates, config.game.defender_action_set)
    # print_action_values(cost_estimates, config.game.defender_action_set)


