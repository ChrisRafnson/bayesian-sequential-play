import time

import filters as jf
import jax
import numpy as np
import pandas as pd
import plotting
from benchmark_policies import FP_policy, get_MC_action
from config import MainConfig
from game_managers import SatelliteOperationGame
from models import SatelliteOperator
from policies import (
    compute_value_functions,
    generate_all_trajectories,
    get_ADP_action_linear,
    get_boltzmann_action,
    get_H2S_action,
    get_means,
)
from utility_functions import get_utility


def evaluate_boltzmann_policy(key, seed, config: MainConfig):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)
    particles = jf.initialize_particles_beta_prior(
        subkey, config, num_particles, num_attacker_actions
    )

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()
        defender_action = get_boltzmann_action(subkey, particles, action_matrix)
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        defender_utility = get_utility(
            defender_betas, defender_observation, defender_action
        )
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        # Finally, we update our particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            defender_observation,
            config,
        )

        # Store particle stats from the updated particles
        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        p5_particle = np.percentile(np.array(particles), 5, axis=0)
        p95_particle = np.percentile(np.array(particles), 95, axis=0)

        particle_means_record.append(mean_particle)
        filter_means_record.append(mean_particle)
        filter_stds_record.append(std_particle)
        filter_p5_record.append(p5_particle)
        filter_p95_record.append(p95_particle)

        results = {
            "policy": "Boltzmann",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def evaluate_reduced_k_boltzmann_policy(key, seed, config: MainConfig):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)
    particles = jf.initialize_particles_beta_prior(
        subkey, config, num_particles, num_attacker_actions
    )

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()
        defender_action = get_boltzmann_action(subkey, particles, action_matrix, k=0.1)
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        defender_utility = get_utility(
            defender_betas, defender_observation, defender_action
        )
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        # Finally, we update our particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            defender_observation,
            config,
        )

        # Store particle stats from the updated particles
        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        p5_particle = np.percentile(np.array(particles), 5, axis=0)
        p95_particle = np.percentile(np.array(particles), 95, axis=0)

        particle_means_record.append(mean_particle)
        filter_means_record.append(mean_particle)
        filter_stds_record.append(std_particle)
        filter_p5_record.append(p5_particle)
        filter_p95_record.append(p95_particle)

        results = {
            "policy": "Boltzmann_Reduced_K",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def evaluate_H2S_policy(key, seed, config: MainConfig, H, trajectories):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)
    particles = jf.initialize_particles_beta_prior(
        subkey, config, num_particles, num_attacker_actions
    )

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()
        defender_action = get_H2S_action(subkey, particles, trajectories, action_matrix, config)
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        defender_utility = get_utility(
            defender_betas, defender_observation, defender_action
        )
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        # Finally, we update our particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            defender_observation,
            config,
        )

        # Store particle stats from the updated particles
        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        p5_particle = np.percentile(np.array(particles), 5, axis=0)
        p95_particle = np.percentile(np.array(particles), 95, axis=0)

        particle_means_record.append(mean_particle)
        filter_means_record.append(mean_particle)
        filter_stds_record.append(std_particle)
        filter_p5_record.append(p5_particle)
        filter_p95_record.append(p95_particle)

        results = {
            "policy": f"H2S (H={H})",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def evaluate_AMG_policy(key, seed, config: MainConfig):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)
    particles = jf.initialize_particles_beta_prior(
        subkey, config, num_particles, num_attacker_actions
    )

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Generate trajectories for AMG (H2S = 0)
    trajectories = generate_all_trajectories(0)

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()
        defender_action = get_H2S_action(subkey, particles, trajectories, action_matrix, config)
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        defender_utility = get_utility(
            defender_betas, defender_observation, defender_action
        )
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        # Finally, we update our particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            defender_observation,
            config,
        )

        # Store particle stats from the updated particles
        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        p5_particle = np.percentile(np.array(particles), 5, axis=0)
        p95_particle = np.percentile(np.array(particles), 95, axis=0)

        particle_means_record.append(mean_particle)
        filter_means_record.append(mean_particle)
        filter_stds_record.append(std_particle)
        filter_p5_record.append(p5_particle)
        filter_p95_record.append(p95_particle)

        results = {
            "policy": "AMG",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def evaluate_ADP_policy(key, seed, config: MainConfig, model_matrix):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_defender_actions = int(
        config.game.num_defender_actions
    )  # This needs to be a concrete int for some reason, in order to compile the get_ADP_action_linear function
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)
    particles = jf.initialize_particles_beta_prior(
        subkey, config, num_particles, num_attacker_actions
    )

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()
        mean_particle = get_means(particles)
        defender_action = get_ADP_action_linear(
            mean_particle, model_matrix[timestep], num_defender_actions, action_matrix
        )
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        defender_utility = get_utility(
            defender_betas, defender_observation, defender_action
        )
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        # Finally, we update our particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            defender_observation,
            config,
        )

        # Store particle stats from the updated particles
        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        p5_particle = np.percentile(np.array(particles), 5, axis=0)
        p95_particle = np.percentile(np.array(particles), 95, axis=0)

        particle_means_record.append(mean_particle)
        filter_means_record.append(mean_particle)
        filter_stds_record.append(std_particle)
        filter_p5_record.append(p5_particle)
        filter_p95_record.append(p95_particle)

        results = {
            "policy": "ADP",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def evaluate_MC_policy(key, seed, config: MainConfig):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()

        defender_action = get_MC_action(model, manager, config)
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        # defender_utility = get_utility(defender_betas, defender_observation, defender_action)
        defender_utility = get_utility(defender_betas, latent_state, defender_action)
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        results = {
            "policy": "MC",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def evaluate_FP_policy(key, seed, config: MainConfig):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_timesteps = config.game.num_timesteps
    action_matrix = config.game.action_matrix
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    key, subkey = jax.random.split(key)

    manager = SatelliteOperationGame(config)
    model = SatelliteOperator(config)

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    latent_state_record = []
    observation_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    # Defender Model
    defender_model = FP_policy(config)

    # Play the satellite game
    for timestep in range(num_timesteps):

        # Determine player actions
        start_time = time.perf_counter()
        defender_action = defender_model.select_action()
        end_time = time.perf_counter()
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # Iterate the game forward
        latent_state, defender_observation, attacker_observation = manager.step(
            attacker_action, defender_action
        )
        latent_state_record.append(latent_state)
        observations = [attacker_observation, defender_observation]
        observation_record.append(observations)

        # The opponent's model is then updated
        model.update_model(defender_action, attacker_observation)

        # Store the utilities given to each player
        attacker_utility = get_utility(
            attacker_betas, attacker_observation, attacker_action
        )
        defender_utility = get_utility(
            defender_betas, defender_observation, defender_action
        )
        utilities = [attacker_utility, defender_utility]
        utility_record.append(utilities)

        results = {
            "policy": "FP",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "latent_states": latent_state_record,
            "observations": observation_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": attacker_betas,
            "defender_betas": defender_betas,
        }

    return results


def compute_episode_summary(results: dict, config: MainConfig):

    utility_record = results["utilities"]
    policy_compute_time_record = results["compute_times"]
    latent_state_record = results["latent_states"]
    action_record = np.array(results["actions"])

    # ----------------------------Utility Related Statistics-----------------------------------------

    # Average and Total Util

    UR = np.array(utility_record)
    attacker_utilities = UR[:, 0]
    defender_utilities = UR[:, 1]

    atk_avg_util = np.average(attacker_utilities)
    def_avg_util = np.average(defender_utilities)

    atk_tot_util = np.sum(attacker_utilities)
    def_tot_util = np.sum(defender_utilities)

    # Window-based Util
    window_size = 10
    UR = np.array(results["utilities"])  # shape (T,2)
    df = pd.DataFrame(UR, columns=["attacker", "defender"])
    df["step"] = np.arange(len(df))
    df["window_id"] = df["step"] // window_size
    win_df = df.groupby("window_id")[["attacker", "defender"]].mean().reset_index()
    win_df["advantage"] = win_df["defender"] - win_df["attacker"]

    # ----------------------------Action Related Statistics-----------------------------------------

    # --- Action distributions ---
    T = action_record.shape[0]
    attacker_actions = action_record[:, 0]
    defender_actions = action_record[:, 1]

    if config is not None:
        atk_counts = np.bincount(
            attacker_actions, minlength=config.game.num_attacker_actions
        )
        def_counts = np.bincount(
            defender_actions, minlength=config.game.num_defender_actions
        )
    else:
        atk_counts = np.bincount(attacker_actions)
        def_counts = np.bincount(defender_actions)

    atk_dist = (atk_counts / T).tolist()
    def_dist = (def_counts / T).tolist()

    # joint distribution
    max_a, max_d = atk_counts.size, def_counts.size
    joint_counts = np.zeros((max_a, max_d), dtype=int)
    for a, d in action_record:
        joint_counts[a, d] += 1
    joint_dist = (joint_counts / T).tolist()

    # ----------------------------Latent State Related Statistics-----------------------------------------
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )
    attacker_betas = np.array(
        [
            config.model.beta_MSD,
            config.model.beta_MAD,
            config.model.beta_C,
            config.model.beta_penalty,
        ]
    )

    mask = (latent_state_record >= defender_betas[0]) & (
        latent_state_record <= defender_betas[1]
    )
    steps_in_defender_bounds = np.sum(mask)  # or mask.sum()

    mask = (latent_state_record >= attacker_betas[0]) & (
        latent_state_record <= attacker_betas[1]
    )
    steps_in_attacker_bounds = np.sum(mask)  # or mask.sum()

    time_spent_in_defender_bounds = steps_in_defender_bounds / len(latent_state_record)
    time_spent_in_attacker_bounds = steps_in_attacker_bounds / len(latent_state_record)

    # ----------------------------Computing Time Related Statistics-----------------------------------------
    PCTR = np.array(policy_compute_time_record)
    avg_comp_time = np.average(PCTR)

    episode_summary = {
        "policy": results["policy"],
        "avg_comp_time": avg_comp_time,
        "atk_avg_util": atk_avg_util,
        "atk_tot_util": atk_tot_util,
        "def_avg_util": def_avg_util,
        "def_tot_util": def_tot_util,
        "atk_action_dist": atk_dist,
        "def_action_dist": def_dist,
        "joint_action_dist": joint_dist,
        "time_in_def_bounds": time_spent_in_defender_bounds,
        "time_in_atk_bounds": time_spent_in_attacker_bounds,
        "window_size": window_size,
        "window_ids": win_df["window_id"].tolist(),
        "atk_win_means": win_df["attacker"].tolist(),
        "def_win_means": win_df["defender"].tolist(),
        "adv_win_means": win_df["advantage"].tolist(),
        "attacker_actions": np.array(attacker_actions),
        "defender_actions": np.array(defender_actions),
        "latent_state_record": np.array(latent_state_record),
        "attacker_betas": np.array(attacker_betas),
        "defender_betas": np.array(defender_betas),
        "attacker_utilities": np.array(attacker_utilities),
        "defender_utilities": np.array(defender_utilities),
    }

    return episode_summary


def print_episode_summary(episode_summary: dict, withHeader=True):
    policy_name = episode_summary["policy"]
    avg_comp_time = episode_summary["avg_comp_time"]
    atk_avg_util = episode_summary["atk_avg_util"]
    atk_tot_util = episode_summary["atk_tot_util"]
    def_avg_util = episode_summary["def_avg_util"]
    def_tot_util = episode_summary["def_tot_util"]

    # Define column headers and widths
    headers = [
        "Policy",
        "Avg Comp Time",
        "Atk Avg Util",
        "Atk Tot Util",
        "Def Avg Util",
        "Def Tot Util",
    ]
    values = [
        policy_name,
        avg_comp_time,
        atk_avg_util,
        atk_tot_util,
        def_avg_util,
        def_tot_util,
    ]
    widths = [15, 15, 15, 15, 15, 15]

    # Format values: if numeric → 4 decimals
    def fmt(val):
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)

    formatted_values = [fmt(v) for v in values]

    if withHeader:
        # Print header
        header_row = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
        print(header_row)
        print("-" * len(header_row))

    # Print values
    value_row = "".join(f"{v:<{w}}" for v, w in zip(formatted_values, widths))
    print(value_row)


if __name__ == "__main__":

    key = jax.random.key(42)
    seed = 42

    key, subkey = jax.random.split(key)
    config = MainConfig.create(subkey)

    # results = evaluate_boltzmann_policy(key, seed, config)
    results = evaluate_H2S_policy(key, seed, config, H=1)
    # results = evaluate_AMG_policy(key, seed, config)

    # key, subkey = jax.random.split(key)
    # start = time.perf_counter()
    # model_matrix = compute_value_functions(subkey, config, config.game.num_timesteps)
    # end = time.perf_counter()
    # print(f"Time to compute value functions: {end - start:.4f} seconds")
    # results = evaluate_ADP_policy(key, seed, config, model_matrix)

    episode_summary = compute_episode_summary(results)
    print_episode_summary(episode_summary)
