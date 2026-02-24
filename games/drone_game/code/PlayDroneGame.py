import os

print(os.getcwd())

import time
import pandas as pd
import filters as jf
import jax
import jax.numpy as jnp
import models
import numpy as np
import plotting as plotting
from config import MainConfig
from benchmark_policies import get_MC_action, get_MC_action_jax, FP_policy, DirichletMultinomial
from policies import (
    generate_all_trajectories,
    get_boltzmann_action,
    get_H2S_action,
    get_means,
)
from utility_functions import attacker_utility, defender_utility
from constants import (
    LOGICAL_PARTICLE_DIM
)
from utils import index_to_ij
from ADP import compute_value_functions, get_ADP_action_MLP
import regressions as rf

seed = 1

np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
config = MainConfig.create(subkey, game_kwargs={"num_sites": 8, "num_attacker_selections": 1})

# Start the EWA model
model = models.DroneAttacker(config)
def_model = DirichletMultinomial(1, config)

# Intialize particles

start_time = time.perf_counter()
key, subkey = jax.random.split(key)
num_particles = config.filter.num_particles
num_attacker_actions = config.game.num_attacker_actions
num_defender_actions = int(config.game.num_defender_actions)
particles = jf.initialize_particles_beta_prior(
    subkey, config, num_particles, num_attacker_actions
)
end_time = time.perf_counter()

print(f"Time to Initialize Particles: {end_time - start_time:.6f} seconds")

num_steps = config.game.num_timesteps

time_per_step = []
filter_data = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # D = number of dimensions in your particles
filter_stds = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # same shape as means
filter_mins = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # D = number of dimensions in your particles
filter_maxs = np.zeros((num_steps, LOGICAL_PARTICLE_DIM))  # D = number of dimensions in your particles

H=3

trajectories = generate_all_trajectories(H, config)

records = []

# # Generate models
# print("Training ADP Models...")
# key, subkey = jax.random.split(key)
# model_seq = compute_value_functions(subkey, config, num_steps)

# # Save models
# rf.save_params_sequence([m.params for m in model_seq], "models/mlp_sequence.msgpack")
# model_seq = rf.load_params_sequence("/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_ADP_models/A2_ADP_model_no_priorities_decision_every_epoch_v1.msgpack", num_steps)

defender_actions = jnp.arange(num_defender_actions)

start = time.time()
for i in range(num_steps):

    mean_particle = get_means(particles)
    std_particle = jf.get_stds(particles)
    min_particle = jf.get_mins(particles)
    max_particle = jf.get_maxs(particles)

    start_time = time.perf_counter()

    # The first step is to get the actions
    key, subkey = jax.random.split(key)
    defender_action = get_boltzmann_action(subkey, particles, defender_actions, config)
    # defender_action = get_H2S_action(subkey, particles, trajectories, config)
    # defender_action = get_ADP_action_linear(mean_particle, model_matrix[i])
    # defender_action = def_model.select_action()
    # defender_action = get_MC_action_jax(model.probabilities, config, i)
    # defender_action = get_ADP_action_MLP(mean_particle, model_seq[i], defender_actions, config)


    attacker_action = model.select_action()
    actions = jnp.array([attacker_action, defender_action])
    end_time = time.perf_counter()

    # The opponent's model is then updated
    model.update_model(defender_action, i)

    # # The defender's model is then updated
    # def_model.update_model(attacker_action)

    # Finally, we update our particle filter
    key, subkey = jax.random.split(key)
    start_time_filter = time.perf_counter()
    particles = jf.step(
        subkey,
        particles,
        attacker_action,
        defender_action,
        config,
    )
    end_time_filter = time.perf_counter()

    filter_data[i] = np.array(mean_particle)
    filter_stds[i] = np.array(std_particle)
    filter_mins[i] = np.array(min_particle)
    filter_maxs[i] = np.array(max_particle)

    # np.savetxt('filter_output.csv', particles, delimiter=',')

    sites = index_to_ij(actions[0])
    a, b = map(int, sites)  

    records.append({"t": i, "site": a, "role": "Attack"})
    records.append({"t": i, "site": b, "role": "Attack"})
    records.append({"t": i, "site": int(actions[1]),   "role": "Defense"})

    print(
        f"Sites Attacked: [{a:2d}, {b:2d}] | "
        f"Site Undefended: {actions[1]:2d} | "
        f"Attacker Payoff: {attacker_utility(attacker_action, defender_action, model.beta, config, timestep=i):8.4f} | "
        f"Defender Payoff: {defender_utility(attacker_action, defender_action, config, timestep=i):8.4f} | "
        f"Time to Compute Action: {end_time - start_time:9.6f} s | "
        f"Time to Compute Particles: {end_time_filter - start_time_filter:9.6f} s"
    )


# jf.print_particle_with_labels(jf.get_means(particles))

np.savetxt("filter_output_last.csv", particles, delimiter=",")


def consolidate_model_data(
    latent_state_record,
    experience_record,
    attraction_record,
    probability_record,
    config: MainConfig,
):
    """
    Consolidates model data into a single matrix.
    Each row corresponds to a timestep.
    Columns are ordered as follows:
    [delta, phi, rho, lambda, N, beta_1, beta_2, beta_3, X, eta,
     A0, A1, A2, P0, P1, P2]

    Args:
        latent_state_record (list): list of X over timesteps
        eta (float): constant eta value
        EWA_params (dict): parameters dict with keys: delta, phi, rho, lambda, beta_1, beta_2, beta_3
        experience_record (list): list of N over timesteps
        attraction_record (list of lists): list of [A0,A1,A2] over timesteps
        probability_record (list of lists): list of [P0,P1,P2] over timesteps

    Returns:
        np.ndarray: matrix of shape (n_timesteps, 16)
    """
    # number of timesteps
    n_timesteps = len(latent_state_record)

    # extract EWA parameters in correct order
    params = np.array(
        [
            config.model.delta,
            config.model.phi,
            config.model.rho,
            config.model.Lambda,
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    # replicate params & eta across all timesteps
    params_tiled = np.tile(params, (n_timesteps, 1))
    eta_column = np.full(
        (n_timesteps, 1),
        config.game.error_Z_variance,
    )

    # convert per-timestep records to arrays
    experience = np.array(experience_record).reshape(-1, 1)
    latent_state = np.array(latent_state_record).reshape(-1, 1)
    attractions = np.array(attraction_record)  # shape: (n,3)
    probabilities = np.array(probability_record)  # shape: (n,3)

    # construct the final matrix column by column
    matrix = np.hstack(
        [
            params_tiled[:, 0:1],  # delta
            params_tiled[:, 1:2],  # phi
            params_tiled[:, 2:3],  # rho
            params_tiled[:, 3:4],  # lambda
            experience,  # N
            params_tiled[:, 4:5],  # beta_1
            params_tiled[:, 5:6],  # beta_2
            params_tiled[:, 6:7],  # beta_3
            params_tiled[:, 7:8],  # beta_3
            latent_state,  # X
            eta_column,  # eta
            attractions[:, 0:1],  # A0
            attractions[:, 1:2],  # A1
            attractions[:, 2:3],  # A2
            probabilities[:, 0:1],  # P0
            probabilities[:, 1:2],  # P1
            probabilities[:, 2:3],  # P2
        ]
    )

    return matrix[:-1, :]


end = time.time()

print(f"Time Elapsed: {end - start}")


df = pd.DataFrame(records)
df.to_csv("game_log.csv", index=False)

# # Plot the true and filter data of the EWA model

# plotting.plot_means_over_time(model_data, filter_data, filter_stds)
# plotting.plot_latent_and_probs(model_data, filter_data, filter_stds)
# # plotting.plot_means_over_time(model_data, filter_mins)
# # plotting.plot_means_over_time(model_data, filter_maxs)
# # plotting.plot_differences_over_time(model_data, filter_data)
