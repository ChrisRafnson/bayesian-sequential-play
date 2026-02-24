import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from beta_functions import (
    draw_beta,
    draw_beta_AB,
)
from functools import partial
from config import MainConfig
from jax import random
from jax.nn import softmax
from utility_functions import attacker_utility
from utils import safe_softmax
from constants import (
    PARAMETER_INDICES,
    MAX_NUM_ATTACKER_ACTIONS,
    BETA_INDICES,
    LOGICAL_PARTICLE_DIM,
    MAX_PARTICLE_DIM,
    IDX_DELTA,
    IDX_PHI,
    IDX_RHO,
    IDX_LAMBDA,
    IDX_N,
    IDX_A,
    IDX_D,
    IDX_T,
    IDX_BETA,
    IDX_A1,
    IDX_P1
)


"""Helper Functions"""
def update_parameter_columns(row, row_updates):
    return row.at[PARAMETER_INDICES].set(row_updates)

# def update_attraction_columns(row, row_updates, config: MainConfig):
#     return row.at[IDX_A1 : IDX_A1 + config.game.num_attacker_actions].set(row_updates)

# def update_attraction_columns(row, row_updates):
#     # updates a 1D segment starting at IDX_A1, length = len(row_updates)
#     return lax.dynamic_update_slice(row, row_updates, (IDX_A1,))

# def update_action_probabilities(particle, num_atk_actions):
#     # contiguous slices avoid building index arrays
#     attractions = particle[IDX_A1 : IDX_A1 + num_atk_actions]
#     lam = particle[IDX_LAMBDA]
#     new_probs = safe_softmax(attractions, k=lam)  # or softmax(lam*attractions)
#     return lax.dynamic_update_slice(particle, new_probs, (IDX_P1,))

# def update_action_probabilities_all(particles, num_atk_actions):
#     return jax.vmap(update_action_probabilities, in_axes=(0, None))(particles, num_atk_actions)

def update_attraction_columns(row, row_updates, K: int):
    # K must be a Python int so slice bounds are static
    return row.at[IDX_A1 : IDX_A1 + K].set(row_updates)

def _update_action_probabilities(particle, K: int):
    # K must be static here as well
    attractions = particle[IDX_A1 : IDX_A1 + K]
    lam = particle[IDX_LAMBDA]
    new_probs = safe_softmax(attractions, k=lam)
    return particle.at[IDX_P1 : IDX_P1 + K].set(new_probs)

def update_action_probabilities_all(particles, K: int):
    f = partial(_update_action_probabilities, K=K)
    return jax.vmap(f, in_axes=(0,))(particles)

def update_beta_columns(row, row_updates):
    return row.at[BETA_INDICES].set(row_updates)

def softmax_probabilities(attractions, Lambda):
    return softmax(jnp.array(attractions) * Lambda)

def first_nonzero_index(arr):
    return int(jnp.argmax(arr != 0)) if jnp.any(arr != 0) else -1

def get_means(particles):
    return jnp.mean(particles[:, :LOGICAL_PARTICLE_DIM], axis=0)

def get_stds(particles):
    """
    Compute standard deviation of each logical particle dimension.

    Args:
        particles (jnp.ndarray): shape (num_particles, total_dims)

    Returns:
        jnp.ndarray: standard deviations for the first LOGICAL_PARTICLE_DIM dimensions
    """
    return jnp.std(particles[:, :LOGICAL_PARTICLE_DIM], axis=0)

def get_mean(particle):
    return jnp.mean(particle[:LOGICAL_PARTICLE_DIM], axis=0)


def get_mins(particles):
    return jnp.min(particles[:, :LOGICAL_PARTICLE_DIM], axis=0)


def get_maxs(particles):
    return jnp.max(particles[:, :LOGICAL_PARTICLE_DIM], axis=0)

def _masked_softmax(x, mask, k):
    # mask: True for active actions 0..K-1, False for inactive K..MAX-1
    x_masked = jnp.where(mask, x, -jnp.inf)
    # safe_softmax(x, k=...) if you prefer; below is equivalent:
    return jax.nn.softmax(k * x_masked, axis=-1)

def _update_action_probabilities_masked(particle, K: int):
    # Read the FULL fixed block (static bounds)
    atts_full = particle[IDX_A1 : IDX_A1 + MAX_NUM_ATTACKER_ACTIONS]
    lam = particle[IDX_LAMBDA]

    # Build mask of length MAX_ATTACKER_ACTIONS (no dynamic slice)
    # K can be Python int or JAX scalar; both work here
    mask = jnp.arange(MAX_NUM_ATTACKER_ACTIONS) < K

    probs_full = _masked_softmax(atts_full, mask, lam)

    # Write back the FULL fixed prob block (static bounds)
    particle = particle.at[IDX_P1 : IDX_P1 + MAX_NUM_ATTACKER_ACTIONS].set(probs_full)
    return particle

def update_action_probabilities_all(particles, K: int):
    # K can be Python int; if you ever jit this wrapper, you can mark K static.
    return jax.vmap(_update_action_probabilities_masked, in_axes=(0, None))(particles, K)

""" Particle Filter Functions"""
def initialize_particles_beta_prior(
    key, config: MainConfig, num_particles, num_attacker_actions, timestep=0
):
    particles = jnp.zeros((num_particles, LOGICAL_PARTICLE_DIM))

    K = int(num_attacker_actions)

    # Get random keys
    key, subkey = random.split(key)
    subkeys = random.split(subkey, 7)

    # Extract the bounds of our priors
    delta_parameters = (*config.priors.delta_parameters, *config.priors.delta_bounds)
    phi_parameters = (*config.priors.phi_parameters, *config.priors.phi_bounds)
    rho_parameters = (*config.priors.rho_parameters, *config.priors.rho_bounds)
    lambda_parameters = (*config.priors.lambda_parameters, *config.priors.lambda_bounds)
    experience_parameters = (
        *config.priors.experience_parameters,
        *config.priors.experience_bounds,
    )
    attraction_parameters = (
        *config.priors.initial_attraction_parameters,
        *config.priors.initial_attraction_bounds,
    )
    beta_parameters = (
        *config.priors.beta_parameters,
        *config.priors.beta_bounds,
    )

    # Draw the intial values, each entry is column so we define the array along the column axis
    delta_vals = draw_beta_AB(subkeys[0], (num_particles,), *delta_parameters)  # Delta
    phi_vals = draw_beta_AB(subkeys[1], (num_particles,), *phi_parameters)  # Phi
    rho_vals = draw_beta_AB(subkeys[2], (num_particles,), *rho_parameters)  # Rho
    lambda_vals = draw_beta_AB(
        subkeys[3], (num_particles,), *lambda_parameters
    )  # Lambda
    exp_vals = draw_beta_AB(
        subkeys[4], (num_particles,), *experience_parameters
    )  # Experience
    beta_vals = draw_beta_AB(
        subkeys[5], (num_particles,), *beta_parameters
    ) #Betas

    #Create columns of -1 for the latent state variables A and D
    A_vals = jnp.ones(shape=(num_particles,)) * -1
    D_vals = jnp.ones(shape=(num_particles,)) * -1
    T_vals = jnp.ones(shape=(num_particles,)) * timestep # Column of zeros for the initial state or whatever we want the starting state to be

    # Draw the intial values, each entry is column so we define the array along the column axis
    initial_values = jnp.stack(
        [
            delta_vals,  # Delta
            phi_vals,  # Phi
            rho_vals,  # Rho
            lambda_vals,  # Lambda
            beta_vals,  # Beta
            exp_vals,  # Experience
            A_vals, #previous attacker action
            D_vals, #previous defender action
            T_vals #Current timestep
        ],
        axis=1,
    )
    particles = particles.at[:, IDX_DELTA : IDX_T + 1].set(
        initial_values
    )  # Insert the intial parameter values into the matrix

    # # Now we draw the attraction values and insert them into the matrix as well.
    # attraction_values = draw_beta_AB(
    #     subkeys[6], (num_particles, num_attacker_actions), *attraction_parameters
    # )
    # particles = jax.vmap(update_attraction_columns)(particles, attraction_values)
    # particles = update_action_probabilities_all(
    #     particles, num_attacker_actions
    # )  # The probabilities of an action are deterministic given the attractions

    # Fill attractions for first K columns; zero elsewhere is fine.
    attraction_values = draw_beta_AB(subkeys[6], (num_particles, K), *attraction_parameters)


    # Write those K columns using dynamic_update_slice (start is static, size is K but write uses full vector if you prefer)
    # Option A: write K via dynamic_update_slice_in_dim
    particles = jax.vmap(
        lambda row, vals: jax.lax.dynamic_update_slice_in_dim(
            row, vals, start_index=IDX_A1, axis=0  # WRONG axis; see below note
        )
    )(particles, attraction_values)

    # Compute probs from attractions
    particles = update_action_probabilities_all(particles, K)

    return particles

"""This applies the jit compilation to the initialization function.
It also tells jax that the number of particles and attacker actions is constant.
The function can support initializing particles for a range of values for the number
of attacker actions, but they likely need to be run as batches..."""
initialize_particles_beta_prior = jax.jit(
    initialize_particles_beta_prior, static_argnums=(2, 3)
)


def _evolve_particle_parameters_single(key, particle, config: MainConfig):

    # Split the key into nine as we need nine draws from a beta
    subkeys = random.split(key, 10)

    AEP_scale = (
        config.priors.AEP_scale
    )  # This is the concentration of the AEP distributions around the current particle values

    # Extract the bounds of our priors
    delta_lower, delta_upper = config.priors.delta_bounds
    phi_lower, phi_upper = config.priors.phi_bounds
    rho_lower, rho_upper = config.priors.rho_bounds
    lambda_lower, lambda_upper = config.priors.lambda_bounds
    beta_lower, beta_upper = config.priors.beta_bounds


    # Evolve delta, phi, rho, lambda, and beta
    parameter_values = jnp.array(
        [
            draw_beta(
                subkeys[0],
                shape=(),
                mean=particle[IDX_DELTA],
                scale=AEP_scale,
                lower_bound=delta_lower,
                upper_bound=delta_upper,
            ),  # Delta
            draw_beta(
                subkeys[1],
                shape=(),
                mean=particle[IDX_PHI],
                scale=AEP_scale,
                lower_bound=phi_lower,
                upper_bound=phi_upper,
            ),  # Phi
            draw_beta(
                subkeys[2],
                shape=(),
                mean=particle[IDX_RHO],
                scale=AEP_scale,
                lower_bound=rho_lower,
                upper_bound=rho_upper,
            ),  # Rho
            draw_beta(
                subkeys[3],
                shape=(),
                mean=particle[IDX_LAMBDA],
                scale=AEP_scale,
                lower_bound=lambda_lower,
                upper_bound=lambda_upper,
            ),  # Lambda
            draw_beta(
                subkeys[4],
                shape=(),
                mean=particle[IDX_BETA],
                scale=AEP_scale,
                lower_bound=beta_lower,
                upper_bound=beta_upper,
            )  # Beta
        ]
    )
    particle = particle.at[IDX_DELTA : IDX_BETA + 1].set(
        parameter_values
    )  # Insert the transitioned values into the matrix

    return particle


def evolve_particle_parameters(key, particles, config: MainConfig):

    num_particles = particles.shape[0]

    # Get keys for the beta values
    keys = random.split(key, num_particles)

    # Apply the new values to the particles
    particles = jax.vmap(_evolve_particle_parameters_single, in_axes=(0, 0, None))(
        keys, particles, config
    )

    return particles


evolve_particle_parameters = jax.jit(evolve_particle_parameters)

def single_attraction_update(a, carry):
    """
    A function to perform a single attraction update for a specific action.


    Args:
        carry (array): Contains the particle, actions taken, the new experience value for the EWA update, the opponent's estimated sensor observation (Z_t), as well as the action matrix.
        a (int): The action/attraction being updated

    Return:
        new_carry (array): Contains the updated row and everything brought in via the original carry variable
    """

    particle, attacker_action, defender_action, config, new_experience = (
        carry  # Unpack the carry
    )

    #Extract particle's timestep value
    timestep = jnp.int16(particle[IDX_T])

    utility = attacker_utility(
        attacker_action, defender_action, particle[IDX_BETA], config, timestep
    )  # Use actions and the beta to get the utility

    # Update the attraction for action i
    new_attraction = (
        (particle[IDX_PHI] * particle[IDX_N] * particle[IDX_A1 + a])
        + (
            particle[IDX_DELTA]
            + (1 - particle[IDX_DELTA])
            # * jax.lax.cond(attacker_action == a, lambda _: 1, lambda _: 0, None) #This method may be slower
            * jnp.where(attacker_action == a, 1, 0)
        )
        * utility
    ) / new_experience
    particle = particle.at[IDX_A1 + a].set(new_attraction)
    return (
        particle,
        attacker_action,
        defender_action,
        config,
        new_experience,
    )


def update_attractions(
    key, particle, attacker_action, defender_action, config: MainConfig
):
    """
    Calculates the new attractions for a particle.

    Args:
        key (PRNG_key): A random key to be used
        particle (array): The particle being updated.
        actions (array): The actions taken.
        action_matrix (array): An array that represents the change in X from a (attacker, defender) action pair

    Returns
        updated_particle (array): The updated particle
    """

    num_actions = config.game.num_attacker_actions


    # Now we perform the EWA updates
    new_experience = particle[IDX_RHO] * particle[IDX_N] + 1
    carry = (
        particle,
        attacker_action,
        defender_action,
        config,
        new_experience,
    )
    updated_particle, *_ = jax.lax.fori_loop(
        0, num_actions, single_attraction_update, carry
    )
    updated_particle = particle.at[IDX_N].set(new_experience)
    return updated_particle


def update_attractions_all(
    key, particles, attacker_action, defender_action, config: MainConfig
):
    num_particles = particles.shape[0]

    keys = random.split(key, num_particles)
    particles = jax.vmap(update_attractions, in_axes=(0, 0, None, None, None))(
        keys, particles, attacker_action, defender_action, config
    )
    return particles


update_attractions_all = jax.jit(update_attractions_all)

def transition_latent_states(particles, attacker_action, defender_action):

    """
    If attacker_action and defender_action are both -1:
        Set them to previous attacker and defender actions (scalars).
    Otherwise:
        Set them to -1.
    Applies to all particles.
    """

    curr_atk_state = particles[:, IDX_A] #Attacker portion of the latent states
    curr_def_state = particles[:, IDX_D] #Defender portion of the latent states
    curr_timesteps = particles[:, IDX_T] #Particle timestep values

    # Boolean condition (scalar)
    condition = jnp.logical_and(curr_atk_state == -1, curr_def_state == -1) #Applies to all particles

    # use elementwise where
    new_attacker = jnp.where(condition, attacker_action, -1) #Set the new attacker action
    new_defender = jnp.where(condition, defender_action, -1) #Set the new defender action
    new_timestep = curr_timesteps + 1 #increment the timestep


    # Update all particles (assuming last 2 columns hold attacker/defender actions)
    particles = particles.at[:, IDX_A].set(new_attacker)
    particles = particles.at[:, IDX_D].set(new_defender)
    particles = particles.at[:, IDX_T].set(new_timestep)

    return particles

transition_latent_states = jax.jit(transition_latent_states)

def _transition_latent_state(particle, attacker_action, defender_action):

    """
    If attacker_action and defender_action are both -1:
        Set them to previous attacker and defender actions (scalars).
    Otherwise:
        Set them to -1.
    Applies to all particles.

    This is primarily for use within the policy functions
    """

    curr_atk_state = particle[IDX_A] #Attacker portion of the latent states
    curr_def_state = particle[IDX_D] #Defender portion of the latent states
    curr_timesteps = particle[IDX_T] #Particle timestep values

    # Boolean condition (scalar)
    condition = jnp.logical_and(curr_atk_state == -1, curr_def_state == -1) #Applies to all particles

    # use elementwise where
    new_attacker = jnp.where(condition, attacker_action, -1) #Set the new attacker action
    new_defender = jnp.where(condition, defender_action, -1) #Set the new defender action
    new_timestep = curr_timesteps + 1 #increment the timestep


    # Update all particles (assuming last 2 columns hold attacker/defender actions)
    particle = particle.at[IDX_A].set(new_attacker)
    particle = particle.at[IDX_D].set(new_defender)
    particle = particle.at[IDX_T].set(new_timestep)

    return particle

transition_latent_states = jax.jit(transition_latent_states)


# def update_action_probabilities(particle, attraction_indices, probability_indices):
#     attractions = particle[attraction_indices]
#     Lambda = particle[IDX_LAMBDA]

#     # new_probabilities = softmax(attractions * Lambda)
#     new_probabilities = safe_softmax(attractions, k=Lambda)
#     particle = particle.at[probability_indices].set(new_probabilities)
#     return particle


# def update_action_probabilities_all(particles, config: MainConfig):
#     num_attacker_actions = config.game.num_attacker_actions
#     attraction_indices = jnp.arange(IDX_A1, IDX_A1 + num_attacker_actions) #Tells JAX where to update attractions
#     probability_indices = jnp.arange(IDX_P1, IDX_A1 + num_attacker_actions)# Tells JAX where to update the probabilities

#     particles = jax.vmap(update_action_probabilities)(particles)
#     return particles


update_action_probabilities_all = jax.jit(update_action_probabilities_all)


# def get_probability_observation(particle, observation):
#     """
#     Here we are finding P(Y_t | X_t) for the estimated X_t within the given particle, which is equivalent to P(error_Y) = P(Y_t - X_t).
#     We also know that error_Y ~ N(0, 3), so P(error_Y) = f(error_Y) where f is the pdf of N(0, 3).

#     Args:
#         particle (array): The particle for which we are calculating the probability of observing Y_t given its estimate of the latent state
#         observation (float): The observed distance between satellites

#     Returns:
#         probability (float): The probability of observing Y_t given the estimate of X_t
#     """

#     error_Y = observation - particle[IDX_X]
#     probability = norm_pdf(error_Y, 0, 3)
#     return probability


def get_probability_action(particle, attacker_action):
    """
    This function returns the probability of the attackers observed action given the attractions of the PREVIOUS step.

    Args:
        particle (array): The particle whose attractions/probabilities we are using to calculate the probability of the attacker's action
        attacker_action (int): The attacker's action

    Returns:
        probability (float): The probability of the attacker playing the observed action under this particles previous probability estimates
    """
    probability = particle[IDX_P1 + attacker_action]
    return probability


def get_joint_probability_particle(particle, attacker_action, defender_observation):
    """
    Gets the joint probability of seeing both the attacker's action and the defender's observation under a specfifc particle's parameters.
    This is also equivalent to the raw weight of the particle without normalization.

    Args:
        particle (array): The particle containing the relevent information
        attacker_action (int): The attacker's observed action

    Returns:
        joint_probability (float): The product of the action probability and the observation probability
    """
    joint_probability = get_probability_action(
        particle, attacker_action
    # ) * get_probability_observation(particle, defender_observation)
    )
    return joint_probability


def get_weights_all(particles, attacker_action):
    """
    Gets the normalized weights for all particles, given an observed attacker action.

    Args:
        particles (array): The array of particles, with each entry being a particle
        attacker_action (int): The attacker's observed action

    Returns:
        norm_weights (array): The normalized weights of all particles
    """

    raw_weights = jax.vmap(get_probability_action, in_axes=(0, None))(
        particles, attacker_action
    )
    norm_weights = normalize(raw_weights)
    return norm_weights


get_weights_all = jax.jit(get_weights_all)


def normalize(weights, eps=1e-12):
    total = jnp.sum(weights)
    # Avoid division by zero
    total = jnp.where(total > 0, total, eps)
    # Normalize
    probs = weights / total
    # Clip to remove tiny negatives from floating point noise
    probs = jnp.clip(probs, 0.0, 1.0)
    # Re-normalize so sum is exactly 1
    return probs / jnp.sum(probs)


def resample_particles(key, particles, weights):
    N = weights.shape[0]
    indices = random.choice(key, N, shape=(N,), p=weights, replace=True)
    return particles[indices], indices

def step(
    key,
    particles,
    attacker_action,
    defender_action,
    config: MainConfig,
):
    """
    Executes a full state transition according to Algorithm 1.
    """

    # Push the particles forward in time, updating the latent state. (Steps 3 and 4)
    particles = transition_latent_states(
        particles, attacker_action, defender_action
    )

    # Calculate normalized weights (Steps 5)
    weights = get_weights_all(particles, attacker_action)

    # Resample particles according to the normalized weights (Step 7)
    key, subkey = random.split(key)
    particles, indices = resample_particles(subkey, particles, weights)

    # Update the attractions and probabilities via EWA update (Steps 9-11)
    key, subkey = random.split(key)
    particles = update_attractions_all(
        subkey, particles, attacker_action, defender_action, config
    )
    particles = update_action_probabilities_all(particles, config.game.num_attacker_actions)

    # Evolve the particle parameters (Steps 12 and 13)
    key, subkey = random.split(key)
    particles = evolve_particle_parameters(subkey, particles, config)

    return particles


step = jax.jit(step)


def check_particles(particles, config: MainConfig):
    K = int(config.game.num_attacker_actions)
    probs = particles[:, IDX_P1 : IDX_P1 + K]

    assert jnp.all(probs >= 0), "Probabilities contain negative values!"
    assert jnp.allclose(
        jnp.sum(probs, axis=1), 1.0, atol=1e-5
    ), "Probabilities do not sum to 1!"


def print_particle_with_labels(particle, config: MainConfig, max_show=15):
    """
    Pretty-print a particle with labeled entries.
    Automatically adapts to the number of attacker actions in config.
    Only shows up to `max_show` entries for long sections (attractions/probs).
    """
    K = int(config.game.num_attacker_actions + 5)
    print("-" * 50)

    # --- Core parameters ---
    param_labels = ["DELTA", "PHI", "RHO", "LAMBDA", "N"]
    for idx, label in zip(PARAMETER_INDICES, param_labels):
        print(f"{label:<8} (idx={idx}): {particle[idx]}")

    # --- Latent States ---

    latent_state_labels = ["Attacker Action", "Defender Action"]
    for idx, label in zip([IDX_A, IDX_D], latent_state_labels):
        print(f"{label:<8} (idx={idx}): {particle[idx]}")

    # --- Beta ---
    print(f"{'BETA':<8} (idx={BETA_INDICES[0]}): {particle[BETA_INDICES[0]]}")

    # --- Attractions ---
    print("\nAttractions:")
    start_a = IDX_A1
    for i in range(min(max_show, K)):
        idx = start_a + i
        print(f"  A{i+1:<3} (idx={idx}): {particle[idx]}")
    if K > max_show:
        print("  ...")

    # --- Probabilities ---
    print("\nProbabilities:")
    start_p = IDX_P1
    for i in range(min(max_show, K)):
        idx = start_p + i
        print(f"  P{i+1:<3} (idx={idx}): {particle[idx]}")
    if K > max_show:
        print("  ...")

    print("-" * 50)



if __name__ == "__main__":

    key = random.PRNGKey(np.random.randint(0, 100))

    attacker_action = 7
    defender_action = 6

    key, subkey = random.split(key)
    config = MainConfig.create(subkey)

    key, subkey = random.split(key)
    particles = initialize_particles_beta_prior(
        key, config, config.filter.num_particles, config.game.num_attacker_actions
    )
    print_particle_with_labels(get_means(particles), config)

    particles = transition_latent_states(particles, attacker_action, defender_action)
    print_particle_with_labels(get_means(particles), config)

    particles = transition_latent_states(particles, attacker_action, defender_action)
    print_particle_with_labels(get_means(particles), config)

    #Calculate normalized weights (Steps 5)
    weights = get_weights_all(particles, attacker_action)

    #Resample particles according to the normalized weights (Step 7)
    key, subkey = random.split(key)
    particles, indices = resample_particles(subkey, particles, weights)

    #Update the attractions and probabilities via EWA update (Steps 9-11)
    key, subkey = random.split(key)
    particles = update_attractions_all(subkey, particles, attacker_action, defender_action, config)
    particles = update_action_probabilities_all(particles, config.game.num_attacker_actions)

    # Evolve the particle parameters (Steps 12 and 13)
    key, subkey = random.split(key)
    particles = evolve_particle_parameters(subkey, particles, config)
    print_particle_with_labels(get_means(particles), config)
