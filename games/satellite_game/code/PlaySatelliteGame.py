import os

print(os.getcwd())

import time

import filters as jf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import models
import numpy as np
import plotting as plotting
from benchmark_policies import FP_policy, get_MC_action
from config import MainConfig
from game_managers import SatelliteOperationGame
from policies import (
    compute_value_functions,
    generate_all_trajectories,
    get_ADP_action_linear,
    get_boltzmann_action,
    get_H2S_action,
    get_means,
)
from utility_functions import get_utility

seed = 2

np.random.seed(seed)  # Set seed
key = jax.random.key(seed)
key, subkey = jax.random.split(key)
config = MainConfig.create(subkey)

# Start up a game manager
manager = SatelliteOperationGame(config)
print(f"Current Distance between satellites: {manager.state}")

# Start the EWA model
model = models.SatelliteOperator(config)
defender_model = FP_policy(config)

print(config.game.action_matrix)

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

# jf.print_particle_with_labels(jf.get_means(particles))

defender_betas = jnp.array(
    [
        config.game.defender_beta_MSD,
        config.game.defender_beta_MAD,
        config.game.defender_beta_C,
        config.game.defender_beta_penalty,
    ]
)

num_steps = config.game.num_timesteps

time_per_step = []
filter_data = np.zeros((num_steps, 17))  # D = number of dimensions in your particles
filter_stds = np.zeros((num_steps, 17))  # same shape as means
filter_mins = np.zeros((num_steps, 17))  # D = number of dimensions in your particles
filter_maxs = np.zeros((num_steps, 17))  # D = number of dimensions in your particles

# #Generate value functions
# key, subkey = jax.random.split(key)
# model_matrix = compute_value_functions(subkey, config, num_steps)

# #Save models
# model_np = np.array(model_matrix)
# np.save("model_matrix_OLS.npy", model_np)

# # Load models
# model_matrix = np.load("model_matrix_OLS.npy")
# model_matrix = jnp.array(model_matrix)

state_record = []

action_matrix = config.game.action_matrix

trajectories = generate_all_trajectories(H=1)

start = time.time()
for i in range(num_steps):

    mean_particle = get_means(particles)
    std_particle = jf.get_stds(particles)
    min_particle = jf.get_mins(particles)
    max_particle = jf.get_maxs(particles)

    start_time = time.perf_counter()
    # The first step is to get the actions
    key, subkey = jax.random.split(key)
    defender_action = get_H2S_action(subkey, particles, trajectories, action_matrix, config)
    # defender_action = get_boltzmann_action(subkey, particles, action_matrix)
    # defender_action = get_ADP_action_linear(
    #     mean_particle, model_matrix[i], num_defender_actions, action_matrix
    # )
    # defender_action = get_MC_action(model, manager, config)
    # defender_action = defender_model.select_action()
    attacker_action = model.select_action()
    actions = jnp.array([attacker_action, defender_action])
    end_time = time.perf_counter()

    # A round of the game is then played
    _, defender_observation, attacker_observation = manager.step(
        attacker_action, defender_action
    )

    # The opponent's model is then updated
    model.update_model(defender_action, attacker_observation)
    # defender_model.update_model(attacker_action, defender_observation)

    # Finally, we update our particle filter

    key, subkey = jax.random.split(key)
    start_time_filter = time.perf_counter()
    particles = jf.step(
        subkey,
        particles,
        attacker_action,
        defender_action,
        defender_observation,
        config,
    )
    end_time_filter = time.perf_counter()

    filter_data[i] = np.array(mean_particle)
    filter_stds[i] = np.array(std_particle)
    filter_mins[i] = np.array(min_particle)
    filter_maxs[i] = np.array(max_particle)

    # np.savetxt('filter_output.csv', particles, delimiter=',')

    formatted_attractions = [f"{val:.4f}" for val in model.attractions]
    print(
        f"Actions Played: {actions} | "
        f"Attacker Payoff: {model.utility_function(attacker_observation, attacker_action):.4f} | "
        f"Defender Payoff: {get_utility(defender_betas, defender_observation, defender_action):.4f} | "
        f"Opponent Attractions: {formatted_attractions} | "
        f"Distance Between Satellites: {manager.state:.4f} | "
        f"Time to Compute Action: {end_time - start_time:.6f} seconds | "
        f"Time to Compute Particles: {end_time_filter - start_time_filter:.6f} seconds"
    )

jf.print_particle_with_labels(jf.get_means(particles))

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

plt.plot(manager.state_record)
plt.axhline(y=config.game.defender_beta_MSD, color="b", linestyle="--", linewidth=2)
plt.axhline(y=config.game.defender_beta_MAD, color="b", linestyle="--", linewidth=2)
plt.axhline(y=config.model.beta_MSD, color="r", linestyle="--", linewidth=2)
plt.axhline(y=config.model.beta_MAD, color="r", linestyle="--", linewidth=2)
plt.ylim(-20, 120)
plt.savefig("game_results.png")

model_data = np.array(
    consolidate_model_data(
        manager.state_record,
        model.experience_record,
        model.attraction_record,
        model.probability_record,
        config,
    )
)
filter_data = np.array(filter_data)


# Plot the true and filter data of the EWA model

plotting.plot_means_over_time(model_data, filter_data, filter_stds)
plotting.plot_latent_and_probs(model_data, filter_data, filter_stds)
# plotting.plot_means_over_time(model_data, filter_mins)
# plotting.plot_means_over_time(model_data, filter_maxs)
# plotting.plot_differences_over_time(model_data, filter_data)
