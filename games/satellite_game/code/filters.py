import jax
import jax.numpy as jnp
import numpy as np
from beta_functions import (
    draw_beta,
    draw_beta_AB,
    evolve_MSD_given_MAD,
    initial_MSD_sample_given_MAD,
)
from config import MainConfig
from jax import random
from jax.nn import softmax
from jax.scipy.stats.norm import pdf as norm_pdf
from utility_functions import get_utility
from utils import safe_softmax

# Predefine a single "particle"
MAX_PARTICLE_DIM = 32
LOGICAL_PARTICLE_DIM = 17
NUM_PARTICLES = 10000
NUM_ACTIONS = 3

IDX_DELTA = 0
IDX_PHI = 1
IDX_RHO = 2
IDX_LAMBDA = 3
IDX_N = 4
IDX_BETA_1 = 5
IDX_BETA_2 = 6
IDX_BETA_3 = 7
IDX_BETA_4 = 8
IDX_X = 9
IDX_ETA = 10
IDX_A0 = 11
IDX_A1 = 12
IDX_A2 = 13
IDX_P0 = 14
IDX_P1 = 15
IDX_P2 = 16

PARAMETER_INDICES = jnp.array(
    [IDX_DELTA, IDX_PHI, IDX_RHO, IDX_LAMBDA, IDX_N]
)  # Tells JAX where to update the parameters, delta, phi, rho, lambda, and the experience
ATTRACTION_INDICES = jnp.array(
    [IDX_A0, IDX_A1, IDX_A2]
)  # Tells JAX where to update the attractions
PROBABILITY_INDICES = jnp.array(
    [IDX_P0, IDX_P1, IDX_P2]
)  # Tells JAX where to update the probabilities
BETA_INDICES = jnp.array(
    [IDX_BETA_1, IDX_BETA_2, IDX_BETA_3, IDX_BETA_4]
)  # Tells JAX where to update the betas
X_INDICES = jnp.array([IDX_X])  # Tells JAX where to update the latent state
ETA_INDICES = jnp.array([IDX_ETA])  # Tells JAX where to update the latent state

MIN_ETA_RANGE = 0.0
MAX_ETA_RANGE = 10.0


"""Helper Functions"""
def update_parameter_columns(row, row_updates):
    return row.at[PARAMETER_INDICES].set(row_updates)


def update_attraction_columns(row, row_updates):
    return row.at[ATTRACTION_INDICES].set(row_updates)


def update_X_columns(row, row_updates):
    return row.at[X_INDICES].set(row_updates)


def update_beta_columns(row, row_updates):
    return row.at[BETA_INDICES].set(row_updates)


def update_eta_columns(row, row_updates):
    return row.at[ETA_INDICES].set(row_updates)


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

def draw_error_X(key):
    return random.normal(key)  # Returns a randomly sampled value from a standard Normal


def draw_error_Z(key, eta):
    """
    Draws an error value to be added to the defender's observation, in order to approximate the attacker's observation.

    Args:
        key (PRNG_key): A key to be used in sampling
        eta (float): The sampling parameter, in this case it is variance of a normal distribution.

    Returns:
        error (float): The error_Z value
    """
    error_Z = random.normal(key) * eta
    return error_Z


def adjusted_sigmoid(x, beta):
    return 1 / (1 + jnp.exp(-(x - beta)))

""" Particle Filter Functions"""


def initialize_particles_beta_prior(
    key, config: MainConfig, num_particles, num_actions
):

    particles = jnp.zeros((num_particles, MAX_PARTICLE_DIM))

    # Get random keys
    key, subkey = random.split(key)
    subkeys = random.split(subkey, 12)

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
    beta_MSD_parameters = (
        *config.priors.beta_MSD_parameters,
        *config.priors.beta_MSD_bounds,
    )
    beta_MAD_parameters = (
        *config.priors.beta_MAD_parameters,
        *config.priors.beta_MAD_bounds,
    )
    beta_C_parameters = (*config.priors.beta_C_parameters, *config.priors.beta_C_bounds)
    beta_penalty_parameters = (
        *config.priors.beta_penalty_parameters,
        *config.priors.beta_penalty_bounds,
    )
    latent_state_parameters = (
        *config.priors.latent_state_parameters,
        *config.priors.latent_state_bounds,
    )
    eta_alpha, eta_beta = config.priors.eta_parameters

    # Draw the intial values, each entry is column so we define the array along the column axis
    delta_vals = draw_beta_AB(subkeys[0], (num_particles,), *delta_parameters)  # Delta
    phi_vals = draw_beta_AB(subkeys[1], (num_particles,), *phi_parameters)  # Phi
    # rho_vals = draw_beta_AB(subkeys[1], (num_particles,), *phi_parameters)        #Rho
    rho_vals = draw_beta_AB(subkeys[2], (num_particles,), *rho_parameters)  # Rho
    lambda_vals = draw_beta_AB(
        subkeys[3], (num_particles,), *lambda_parameters
    )  # Lambda
    exp_vals = draw_beta_AB(
        subkeys[4], (num_particles,), *experience_parameters
    )  # Experience
    MAD_vals = draw_beta_AB(
        subkeys[5], (num_particles,), *beta_MAD_parameters
    )  # Beta_2
    MSD_vals = initial_MSD_sample_given_MAD(
        subkeys[6], num_particles, MAD_vals, *beta_MSD_parameters
    )  # Beta_1
    C_vals = draw_beta_AB(subkeys[7], (num_particles,), *beta_C_parameters)  # Beta_3
    penalty_vals = draw_beta_AB(
        subkeys[8], (num_particles,), *beta_penalty_parameters
    )  # Beta_4
    latent_vals = draw_beta_AB(
        subkeys[9], (num_particles,), *latent_state_parameters
    )  # Latent State
    eta_vals = 1 / (
        random.gamma(subkeys[10], eta_alpha, shape=(num_particles,)) / eta_beta
    )  # Eta

    # Draw the intial values, each entry is column so we define the array along the column axis
    initial_values = jnp.stack(
        [
            delta_vals,  # Delta
            phi_vals,  # Phi
            rho_vals,  # Rho
            lambda_vals,  # Lambda
            exp_vals,  # Experience
            MSD_vals,  # Beta_2
            MAD_vals,  # Beta_1
            C_vals,  # Beta_3
            penalty_vals,  # Beta_4
            latent_vals,  # Latent State
            eta_vals,  # Eta
        ],
        axis=1,
    )
    particles = particles.at[:, IDX_DELTA : IDX_ETA + 1].set(
        initial_values
    )  # Insert the intial values into the matrix

    # Now we draw the attraction values and insert them into the matrix as well.
    attraction_values = draw_beta_AB(
        subkeys[11], (num_particles, num_actions), *attraction_parameters
    )
    particles = jax.vmap(update_attraction_columns)(particles, attraction_values)
    particles = update_action_probabilities_all(
        particles
    )  # The probabilities of an action are deterministic given the attractions
    return particles


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
    beta_MSD_lower, beta_MSD_upper = config.priors.beta_MSD_bounds
    beta_MAD_lower, beta_MAD_upper = config.priors.beta_MAD_bounds
    beta_C_lower, beta_C_upper = config.priors.beta_C_bounds
    beta_penalty_lower, beta_penalty_upper = config.priors.beta_penalty_bounds

    # Evolve delta, phi, rho, lambda
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
            # draw_beta(subkeys[1], shape=(), mean=particle[IDX_PHI],    scale=AEP_scale, lower_bound=phi_lower,    upper_bound=phi_upper),           #Rho
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
        ]
    )
    particle = particle.at[IDX_DELTA : IDX_LAMBDA + 1].set(
        parameter_values
    )  # Insert the intial values into the matrix

    # Evolve Betas

    new_MAD_val = draw_beta(
        subkeys[4],
        shape=(),
        mean=particle[IDX_BETA_2],
        scale=AEP_scale,
        lower_bound=beta_MAD_lower,
        upper_bound=beta_MAD_upper,
    )  # Beta_2
    new_MSD_Val = evolve_MSD_given_MAD(
        subkeys[5],
        shape=(),
        scale=AEP_scale,
        MSD=particle[IDX_BETA_1],
        MAD=new_MAD_val,
        lower_bound=beta_MSD_lower,
    )
    new_C_val = draw_beta(
        subkeys[6],
        shape=(),
        mean=particle[IDX_BETA_3],
        scale=AEP_scale,
        lower_bound=beta_C_lower,
        upper_bound=beta_C_upper,
    )
    new_penalty_val = draw_beta(
        subkeys[7],
        shape=(),
        mean=particle[IDX_BETA_4],
        scale=AEP_scale,
        lower_bound=beta_penalty_lower,
        upper_bound=beta_penalty_upper,
    )

    beta_values = jnp.array([new_MSD_Val, new_MAD_val, new_C_val, new_penalty_val])
    particle = particle.at[IDX_BETA_1 : IDX_BETA_4 + 1].set(
        beta_values
    )  # Insert the intial values into the matrix

    # Evolve Eta
    eta_value = draw_beta(
        subkeys[8],
        shape=(1,),
        mean=particle[IDX_ETA],
        scale=AEP_scale,
        lower_bound=MIN_ETA_RANGE,
        upper_bound=MAX_ETA_RANGE,
    )
    particle = update_eta_columns(particle, eta_value)
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

def _transition_latent_states_single(
    key, particle, attacker_action, defender_action, action_matrix
):
    """
    Executes a transition of the latent state for a given particle

    Args:
        key (PRNG_key): A key to sample the perturbation error
        particle (array): The single particle containing the necessary data, such as latent state
        defender_action (int): The defender action
        attacker_action (int): The attacker action
        action_matrix (array): An array that represents the change in X from a (attacker, defender) action pair

    Returns:
        new_particle (array): The updated particle including the new latent state
    """
    error_X = draw_error_X(key)  # Sample the perturbation error
    transition = particle[IDX_X] + action_matrix[attacker_action, defender_action]
    new_state = transition + error_X
    new_particle = particle.at[IDX_X].set(new_state)
    return new_particle


def transition_latent_states(
    key, particles, attacker_action, defender_action, config: MainConfig
):
    """
    Executes a transition of the latent state of all particles

    Args:
        key (PRNG_key): A key to sample the perturbation error
        particle (array): The array containing all particles
        actions (tuple): The tuple containing the actions player
        action_matrix (array): An array that represents the change in X from a (attacker, defender) action pair

    Returns:
        new_particles (array): The updated particles
    """

    action_matrix = config.game.action_matrix
    num_particles = particles.shape[0]

    keys = random.split(key, num_particles)
    new_particles = jax.vmap(
        _transition_latent_states_single, in_axes=(0, 0, None, None, None)
    )(keys, particles, attacker_action, defender_action, action_matrix)
    return new_particles


transition_latent_states = jax.jit(transition_latent_states)

def single_attraction_update(a, carry):
    """
    A function to perform a single attraction update for a specific action.

    Notes:
        Because the payoffs are based entirely on the distance between the two satellites, we need a way to evaluate the payoffs for actions we did not take.
        To account for this, we adjust Z_tn by removing the distance added from the two player actions, this is called K_tn. Then as we loop through the attraction updates,
        we add the distance that would have been added if the attacker made action a and the defender made the observed action (K_tna). K_tna is then fed into the utility
        function along with the betas to provide an estimate for the attacker's utility, which is then used to update the attraction for that action a.

    Args:
        carry (array): Contains the particle, actions taken, the new experience value for the EWA update, the opponent's estimated sensor observation (Z_t), as well as the action matrix.
        i (int): The action/attraction being updated

    Return:
        new_carry (array): Contains the updated row and everything brought in via the original carry variable
    """

    particle, attacker_action, defender_action, new_experience, Z_tn, action_matrix = (
        carry  # Unpack the carry
    )
    K_tna = (
        Z_tn
        - action_matrix[attacker_action, defender_action]
        + action_matrix[a, defender_action]
    )  # Adjust Z_tn for the new action. Notice that K_tna = Z_t when the action evaluated is the one taken.
    utility = get_utility(
        particle[BETA_INDICES], K_tna, attacker_action
    )  # Use K_tna and the betas to get the utility

    # Update the attraction for action i
    new_attraction = (
        (particle[IDX_PHI] * particle[IDX_N] * particle[IDX_A0 + a])
        + (
            particle[IDX_DELTA]
            + (1 - particle[IDX_DELTA])
            * jax.lax.cond(attacker_action == a, lambda _: 1, lambda _: 0, None)
        )
        * utility
    ) / new_experience
    particle = particle.at[IDX_A0 + a].set(new_attraction)
    return (
        particle,
        attacker_action,
        defender_action,
        new_experience,
        Z_tn,
        action_matrix,
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
    action_matrix = config.game.action_matrix

    # Sample the error term for the attacker's observation
    error_Z = draw_error_Z(key, particle[IDX_ETA])

    # Calculate Z_tn from the error and the current estimate of the latent state
    Z_tn = particle[IDX_X] + error_Z

    # Now we perform the EWA updates
    new_experience = particle[IDX_RHO] * particle[IDX_N] + 1
    carry = (
        particle,
        attacker_action,
        defender_action,
        new_experience,
        Z_tn,
        action_matrix,
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
    action_matrix = config.game.action_matrix

    keys = random.split(key, num_particles)
    particles = jax.vmap(update_attractions, in_axes=(0, 0, None, None, None))(
        keys, particles, attacker_action, defender_action, config
    )
    return particles


update_attractions_all = jax.jit(update_attractions_all)


def update_action_probabilities(particle):
    attractions = particle[ATTRACTION_INDICES]
    Lambda = particle[IDX_LAMBDA]

    # new_probabilities = softmax(attractions * Lambda)
    new_probabilities = safe_softmax(attractions, k=Lambda)
    particle = particle.at[PROBABILITY_INDICES].set(new_probabilities)
    return particle


def update_action_probabilities_all(particles):
    particles = jax.vmap(update_action_probabilities)(particles)
    return particles


update_action_probabilities_all = jax.jit(update_action_probabilities_all)


def get_probability_observation(particle, observation):
    """
    Here we are finding P(Y_t | X_t) for the estimated X_t within the given particle, which is equivalent to P(error_Y) = P(Y_t - X_t).
    We also know that error_Y ~ N(0, 3), so P(error_Y) = f(error_Y) where f is the pdf of N(0, 3).

    Args:
        particle (array): The particle for which we are calculating the probability of observing Y_t given its estimate of the latent state
        observation (float): The observed distance between satellites

    Returns:
        probability (float): The probability of observing Y_t given the estimate of X_t
    """

    error_Y = observation - particle[IDX_X]
    probability = norm_pdf(error_Y, 0, 3)
    return probability


def get_probability_action(particle, attacker_action):
    """
    This function returns the probability of the attackers observed action given the attractions of the PREVIOUS step.

    Args:
        particle (array): The particle whose attractions/probabilities we are using to calculate the probability of the attacker's action
        attacker_action (int): The attacker's action

    Returns:
        probability (float): The probability of the attacker playing the observed action under this particles previous probability estimates
    """
    probability = particle[IDX_P0 + attacker_action]
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
    ) * get_probability_observation(particle, defender_observation)
    return joint_probability


def get_weights_all(particles, attacker_action, defender_observation):
    """
    Gets the normalized weights for all particles, given an observed attacker action.

    Args:
        particles (array): The array of particles, with each entry being a particle
        attacker_action (int): The attacker's observed action

    Returns:
        norm_weights (array): The normalized weights of all particles
    """

    raw_weights = jax.vmap(get_joint_probability_particle, in_axes=(0, None, None))(
        particles, attacker_action, defender_observation
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
    defender_observation,
    config: MainConfig,
):
    """
    Executes a full state transition according to Algorithm 1.
    """

    # Push the particles forward in time, updating the latent state. (Steps 3 and 4)
    key, subkey = random.split(key)
    particles = transition_latent_states(
        subkey, particles, attacker_action, defender_action, config
    )

    # Calculate normalized weights (Steps 5)
    weights = get_weights_all(particles, attacker_action, defender_observation)

    # Resample particles according to the normalized weights (Step 7)
    key, subkey = random.split(key)
    particles, indices = resample_particles(subkey, particles, weights)

    # Update the attractions and probabilities via EWA update (Steps 9-11)
    key, subkey = random.split(key)
    particles = update_attractions_all(
        subkey, particles, attacker_action, defender_action, config
    )
    particles = update_action_probabilities_all(particles)

    # Evolve the particle parameters (Steps 12 and 13)
    key, subkey = random.split(key)
    particles = evolve_particle_parameters(subkey, particles, config)

    return particles


step = jax.jit(step)


def check_particles(particles):
    probs = particles[:, PROBABILITY_INDICES]
    assert jnp.all(probs >= 0), "Probabilities contain negative values!"
    assert jnp.allclose(
        jnp.sum(probs, axis=1), 1.0, atol=1e-5
    ), "Probabilities do not sum to 1!"


def print_particle_with_labels(particle):
    labels = [
        "DELTA",
        "PHI",
        "RHO",
        "LAMBDA",
        "N",
        "BETA_1",
        "BETA_2",
        "BETA_3",
        "BETA_4",
        "X",
        "ETA",
        "A0",
        "A1",
        "A2",
        "P0",
        "P1",
        "P2",
    ]

    print("-" * 40)
    for idx, label in enumerate(labels):
        print(f"{label:<8} (idx={idx}): {particle[idx]}")
    print("-" * 40)


if __name__ == "__main__":

    key = random.PRNGKey(np.random.randint(0, 100))

    attacker_action = 1
    defender_action = 1

    key, subkey = random.split(key)
    config = MainConfig.create(subkey)

    key, subkey = random.split(key)
    particles = initialize_particles_beta_prior(
        key, config, config.filter.num_particles, config.game.num_attacker_actions
    )
    print_particle_with_labels(get_means(particles))

    # #Push the particles forward in time, updating the latent state. (Steps 3 and 4)
    # key, subkey = random.split(key)
    # particles = transition_latent_states(subkey, particles, attacker_action, defender_action, config)

    # #Calculate normalized weights (Steps 5)
    # defender_observation = get_means(particles)[IDX_X]
    # weights = get_weights_all(particles, attacker_action, defender_observation)

    # #Resample particles according to the normalized weights (Step 7)
    # key, subkey = random.split(key)
    # particles, indices = resample_particles(subkey, particles, weights)

    # #Update the attractions and probabilities via EWA update (Steps 9-11)
    # key, subkey = random.split(key)
    # particles = update_attractions_all(subkey, particles, attacker_action, defender_action, config)
    # particles = update_action_probabilities_all(particles)

    # Evolve the particle parameters (Steps 12 and 13)
    key, subkey = random.split(key)
    particles = evolve_particle_parameters(subkey, particles, config)
    print_particle_with_labels(get_means(particles))
