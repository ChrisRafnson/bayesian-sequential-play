import jax
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
from utility_functions import utility
from jax.scipy.stats.norm import pdf as norm_pdf, logpdf
from jax.scipy.special import logsumexp
from utils import safe_softmax, kinematic_transition, inverse_kinematic_transition
from constants import (
    PARAMETER_INDICES,
    MAX_NUM_ATTACKER_ACTIONS,
    BETA_INDICES,
    LOGICAL_PARTICLE_DIM,
    NUM_PARTICLES,
    IDX_DELTA,
    IDX_PHI,
    IDX_RHO,
    IDX_LAMBDA,
    IDX_N,
    IDX_SIGMA,
    IDX_BETA_A1,
    IDX_BETA_A2,
    IDX_BETA_A3,
    IDX_BETA_A4,
    IDX_BETA_A5,
    IDX_BETA_A6,
    IDX_XD1,
    IDX_XD2,
    IDX_XD3,
    IDX_XD4,
    IDX_XA1,
    IDX_XA2,
    IDX_XA3,
    IDX_XA4,
    IDX_A1,
    IDX_P1
)


"""Helper Functions"""

def effective_sample_size(weights):
    weights = weights / jnp.sum(weights)
    return 1.0 / jnp.sum(weights**2)


def update_parameter_columns(row, row_updates):
    return row.at[PARAMETER_INDICES].set(row_updates)

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
    key, config: MainConfig, num_particles, num_attacker_actions):
    particles = jnp.zeros((num_particles, LOGICAL_PARTICLE_DIM))

    K = int(num_attacker_actions)

    # Get random keys
    # key, subkey = random.split(key)
    # subkeys = random.split(subkey, 7)

    key, subkey = random.split(key)
    subkeys = random.split(subkey, 20 + K)

    # Extract the bounds of our priors
    delta_parameters = (*config.priors.delta_parameters, *config.priors.delta_bounds)
    phi_parameters = (*config.priors.phi_parameters, *config.priors.phi_bounds)
    rho_parameters = (*config.priors.rho_parameters, *config.priors.rho_bounds)
    lambda_parameters = (*config.priors.lambda_parameters, *config.priors.lambda_bounds)
    sigma_parameters = (*config.priors.sigma_parameters, *config.priors.sigma_bounds)
    experience_parameters = (
        *config.priors.experience_parameters,
        *config.priors.experience_bounds,
    )

    # attraction_parameters = (
    #     *config.priors.initial_attraction_parameters,
    #     *config.priors.initial_attraction_bounds,
    # )

    # Broadcast in case you ever use length-1 defaults; if you hardcode length-K, this is a no-op.
    alpha = jnp.broadcast_to(config.priors.initial_attraction_alpha, (K,))
    beta  = jnp.broadcast_to(config.priors.initial_attraction_beta,  (K,))
    lo    = jnp.broadcast_to(config.priors.initial_attraction_lower, (K,))
    hi    = jnp.broadcast_to(config.priors.initial_attraction_upper, (K,))

    beta_A1_parameters = (
        *config.priors.beta_A1_parameters,
        *config.priors.beta_A1_bounds,
    )
    beta_A2_parameters = (
        *config.priors.beta_A2_parameters,
        *config.priors.beta_A2_bounds,
    )
    beta_A3_parameters = (
        *config.priors.beta_A3_parameters,
        *config.priors.beta_A3_bounds,
    )
    beta_A4_parameters = (
        *config.priors.beta_A4_parameters,
        *config.priors.beta_A4_bounds,
    )
    beta_A5_parameters = (
        *config.priors.beta_A5_parameters,
        *config.priors.beta_A5_bounds,
    )
    beta_A6_parameters = (
        *config.priors.beta_A6_parameters,
        *config.priors.beta_A6_bounds,
    )
    XD1_parameters = (*config.priors.XD1_parameters, *config.priors.XD1_bounds)
    XD2_parameters = (*config.priors.XD2_parameters, *config.priors.XD2_bounds)
    XD3_parameters = (*config.priors.XD3_parameters, *config.priors.XD3_bounds)
    XD4_parameters = (*config.priors.XD4_parameters, *config.priors.XD4_bounds)

    XA1_parameters = (*config.priors.XA1_parameters, *config.priors.XA1_bounds)
    XA2_parameters = (*config.priors.XA2_parameters, *config.priors.XA2_bounds)
    XA3_parameters = (*config.priors.XA3_parameters, *config.priors.XA3_bounds)
    XA4_parameters = (*config.priors.XA4_parameters, *config.priors.XA4_bounds)


    # Draw the intial values, each entry is column so we define the array along the column axis
    delta_vals = draw_beta_AB(subkeys[0], (num_particles,), *delta_parameters)  # Delta
    phi_vals = draw_beta_AB(subkeys[1], (num_particles,), *phi_parameters)  # Phi
    rho_vals = draw_beta_AB(subkeys[2], (num_particles,), *rho_parameters)  # Rho
    lambda_vals = draw_beta_AB(
        subkeys[3], (num_particles,), *lambda_parameters
    )  # Lambda
    sigma_vals = draw_beta_AB(
        subkeys[4], (num_particles,), *sigma_parameters
    )  # Eta
    exp_vals = draw_beta_AB(
        subkeys[5], (num_particles,), *experience_parameters
    )  # Experience
    beta_A1_vals = draw_beta_AB(
        subkeys[6], (num_particles,), *beta_A1_parameters
    ) #Beta_A1
    beta_A2_vals = draw_beta_AB(
        subkeys[7], (num_particles,), *beta_A2_parameters
    ) #Beta_A2
    beta_A3_vals = draw_beta_AB(
        subkeys[8], (num_particles,), *beta_A3_parameters
    ) #Beta_A3
    beta_A4_vals = draw_beta_AB(
        subkeys[9], (num_particles,), *beta_A4_parameters
    ) #Beta_A4
    beta_A5_vals = draw_beta_AB(
        subkeys[10], (num_particles,), *beta_A5_parameters
    ) #Beta_A4
    beta_A6_vals = draw_beta_AB(
        subkeys[11], (num_particles,), *beta_A6_parameters
    ) #Beta_A4

    XD1_vals = draw_beta_AB(subkeys[12], (num_particles,), *XD1_parameters)
    XD2_vals = draw_beta_AB(subkeys[13], (num_particles,), *XD2_parameters)
    XD3_vals = draw_beta_AB(subkeys[14], (num_particles,), *XD3_parameters)
    XD4_vals = draw_beta_AB(subkeys[15], (num_particles,), *XD4_parameters)

    XA1_vals = draw_beta_AB(subkeys[16], (num_particles,), *XA1_parameters)
    XA2_vals = draw_beta_AB(subkeys[17], (num_particles,), *XA2_parameters)
    XA3_vals = draw_beta_AB(subkeys[18], (num_particles,), *XA3_parameters)
    XA4_vals = draw_beta_AB(subkeys[19], (num_particles,), *XA4_parameters)


    # Stack the intial values, each entry is column so we define the array along the column axis
    initial_values = jnp.stack(
        [
            exp_vals,
            delta_vals,
            phi_vals,
            rho_vals,
            lambda_vals,
            beta_A1_vals,
            beta_A2_vals,
            beta_A3_vals,
            beta_A4_vals,
            beta_A5_vals,
            beta_A6_vals,
            sigma_vals,
            XD1_vals, 
            XD2_vals, 
            XD3_vals, 
            XD4_vals,
            XA1_vals, 
            XA2_vals, 
            XA3_vals, 
            XA4_vals, 

        ],
        axis=1,
    )
    particles = particles.at[:, IDX_N : IDX_XA4 + 1].set(
        initial_values
    )  # Insert the intial parameter values into the matrix

    # # Fill attractions for first K columns; zero elsewhere is fine.
    # attraction_values = draw_beta_AB(subkeys[17], (num_particles, K), *attraction_parameters)

    # Use the last K keys for attractions
    keys_attr = subkeys[20:]   # shape (K,)

    def draw_one_attraction(key_k, a_k, b_k, lo_k, hi_k):
        # Draw (num_particles,) for attraction k
        u = random.beta(key_k, a_k, b_k, shape=(num_particles,))
        return lo_k + (hi_k - lo_k) * u

    # (K, num_particles) -> (num_particles, K)
    attraction_values = jax.vmap(draw_one_attraction, in_axes=(0, 0, 0, 0, 0))(
        keys_attr, alpha, beta, lo, hi
    ).T


    # Write those K columns using dynamic_update_slice (start is static, size is K but write uses full vector if you prefer)
    # Option A: write K via dynamic_update_slice_in_dim
    particles = jax.vmap(
        lambda row, vals: jax.lax.dynamic_update_slice_in_dim(
            row, vals, start_index=IDX_A1, axis=0
        )
    )(particles, attraction_values)

    # Compute probs from attractions
    particles = update_action_probabilities_all(particles, K)

    return particles

initialize_particles_beta_prior = jax.jit(
    initialize_particles_beta_prior, static_argnums=(2, 3)
)


def _evolve_particle_parameters_single(key, particle, config: MainConfig):

    # Split the key
    subkeys = random.split(key, 16)

    AEP_scale = (
        config.priors.AEP_scale
    )  # This is the concentration of the AEP distributions around the current particle values

    # Extract the bounds of our priors
    delta_lower, delta_upper = config.priors.delta_bounds
    phi_lower, phi_upper = config.priors.phi_bounds
    rho_lower, rho_upper = config.priors.rho_bounds
    lambda_lower, lambda_upper = config.priors.lambda_bounds
    sigma_lower, sigma_upper = config.priors.sigma_bounds
    beta_A1_lower, beta_A1_upper = config.priors.beta_A1_bounds
    beta_A2_lower, beta_A2_upper = config.priors.beta_A2_bounds
    beta_A3_lower, beta_A3_upper = config.priors.beta_A3_bounds
    beta_A4_lower, beta_A4_upper = config.priors.beta_A4_bounds
    beta_A5_lower, beta_A5_upper = config.priors.beta_A5_bounds
    beta_A6_lower, beta_A6_upper = config.priors.beta_A6_bounds
    XD1_lower, XD1_upper = config.priors.XD1_bounds
    XD2_lower, XD2_upper = config.priors.XD2_bounds
    XD3_lower, XD3_upper = config.priors.XD3_bounds
    XD4_lower, XD4_upper = config.priors.XD4_bounds
    XA1_lower, XA1_upper = config.priors.XA1_bounds
    XA2_lower, XA2_upper = config.priors.XA2_bounds
    XA3_lower, XA3_upper = config.priors.XA3_bounds
    XA4_lower, XA4_upper = config.priors.XA4_bounds


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
            
            #Draw Attacker Beta Vars
            draw_beta(
                subkeys[4],
                shape=(),
                mean=particle[IDX_BETA_A1],
                scale=AEP_scale,
                lower_bound=beta_A1_lower,
                upper_bound=beta_A1_upper,
            ),
            draw_beta(
                subkeys[5],
                shape=(),
                mean=particle[IDX_BETA_A2],
                scale=AEP_scale,
                lower_bound=beta_A2_lower,
                upper_bound=beta_A2_upper,
            ),
            draw_beta(
                subkeys[6],
                shape=(),
                mean=particle[IDX_BETA_A3],
                scale=AEP_scale,
                lower_bound=beta_A3_lower,
                upper_bound=beta_A3_upper,
            ),
            draw_beta(
                subkeys[7],
                shape=(),
                mean=particle[IDX_BETA_A4],
                scale=AEP_scale,
                lower_bound=beta_A4_lower,
                upper_bound=beta_A4_upper,
            ),
            draw_beta(
                subkeys[8],
                shape=(),
                mean=particle[IDX_BETA_A5],
                scale=AEP_scale,
                lower_bound=beta_A5_lower,
                upper_bound=beta_A5_upper,
            ),
            draw_beta(
                subkeys[9],
                shape=(),
                mean=particle[IDX_BETA_A6],
                scale=AEP_scale,
                lower_bound=beta_A6_lower,
                upper_bound=beta_A6_upper,
            ),

            #Draw Noise Parameter
            draw_beta(
                subkeys[10],
                shape=(),
                mean=particle[IDX_SIGMA],
                scale=AEP_scale,
                lower_bound=sigma_lower,
                upper_bound=sigma_upper,
            ),  # Sigma

            # #Draw Defender State Vars
            # draw_beta(
            #     subkeys[11],
            #     shape=(),
            #     mean=particle[IDX_XD1],
            #     scale=AEP_scale,
            #     lower_bound=XD1_lower,
            #     upper_bound=XD1_upper,
            # ),
            # draw_beta(
            #     subkeys[12],
            #     shape=(),
            #     mean=particle[IDX_XD2],
            #     scale=AEP_scale,
            #     lower_bound=XD2_lower,
            #     upper_bound=XD2_upper,
            # ),
            # draw_beta(
            #     subkeys[13],
            #     shape=(),
            #     mean=particle[IDX_XD3],
            #     scale=AEP_scale,
            #     lower_bound=XD3_lower,
            #     upper_bound=XD3_upper,
            # ),
            # draw_beta(
            #     subkeys[14],
            #     shape=(),
            #     mean=particle[IDX_XD4],
            #     scale=AEP_scale,
            #     lower_bound=XD4_lower,
            #     upper_bound=XD4_upper,
            # ),
            # #Draw Attacker State Vars
            # draw_beta(
            #     subkeys[15],
            #     shape=(),
            #     mean=particle[IDX_XA1],
            #     scale=AEP_scale,
            #     lower_bound=XA1_lower,
            #     upper_bound=XA1_upper,
            # ),
            # draw_beta(
            #     subkeys[16],
            #     shape=(),
            #     mean=particle[IDX_XA2],
            #     scale=AEP_scale,
            #     lower_bound=XA2_lower,
            #     upper_bound=XA2_upper,
            # ),
            # draw_beta(
            #     subkeys[17],
            #     shape=(),
            #     mean=particle[IDX_XA3],
            #     scale=AEP_scale,
            #     lower_bound=XA3_lower,
            #     upper_bound=XA3_upper,
            # ),
            # draw_beta(
            #     subkeys[18],
            #     shape=(),
            #     mean=particle[IDX_XA4],
            #     scale=AEP_scale,
            #     lower_bound=XA4_lower,
            #     upper_bound=XA4_upper,
            # ),
            

        ]
    )
    particle = particle.at[IDX_DELTA : IDX_SIGMA + 1].set(
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
    # Unpack the carry
    particle, attacker_observation, attacker_action, defender_action, config, new_experience = (
        carry  
    )

    #Ok our first step is to find an approximation of the pre-decision state

    defender_inputs = config.game.defender_action_set[defender_action]

    XD1_pre, XD2_pre, XD3_pre, XD4_pre = inverse_kinematic_transition(
        attacker_observation[0],
        attacker_observation[1],
        attacker_observation[2],
        attacker_observation[3],
        config.game.L,
        config.game.delta_t,
        defender_inputs[0],
        defender_inputs[1]
    )

    attacker_inputs = config.game.attacker_action_set[attacker_action]

    XA1_pre, XA2_pre, XA3_pre, XA4_pre = inverse_kinematic_transition(
        attacker_observation[4],
        attacker_observation[5],
        attacker_observation[6],
        attacker_observation[7],
        config.game.L,
        config.game.delta_t,
        attacker_inputs[0],
        attacker_inputs[1]
    )

    #Extract the acceleration and steering angle associated with the current attraction being updated
    new_attacker_inputs = config.game.attacker_action_set[a]

    #Using the new action for the attacker, we convert to a theoretical post-decision state
    #Note that when the attraction being updated is the same as the true action take, our calculated
    #post-decision state will be the same as the attacker's observation

    XD1_post, XD2_post, XD3_post, XD4_post = kinematic_transition(
        XD1_pre,
        XD2_pre,
        XD3_pre,
        XD4_pre,
        config.game.L,
        config.game.delta_t,
        defender_inputs[0],
        defender_inputs[1]
    )

    XA1_post, XA2_post, XA3_post, XA4_post = kinematic_transition(
        XA1_pre,
        XA2_pre,
        XA3_pre,
        XA4_pre,
        config.game.L,
        config.game.delta_t,
        new_attacker_inputs[0],
        new_attacker_inputs[1]
    )

    #Now that we have a post-decision state for our current attraction, we can calculate the utility we
    #would receive
    attacker_utility = utility(
        particle[IDX_XA2], #The "true" vertical position of the attacker
        XA1_post, #Attacker horizontal
        XA2_post, #Attacker vertical
        XA3_post, #Attacker vertical
        XA4_post, #Attacker speed
        XD1_post, #Attacker's opponent's horizontal A.K.A defender horizontal
        XD2_post, #defender vertical
        XD3_post, #defender vertical
        XD4_post, #defender speed
        a,
        particle[IDX_BETA_A1],
        particle[IDX_BETA_A2],
        particle[IDX_BETA_A3],
        particle[IDX_BETA_A4],
        particle[IDX_BETA_A5],
        particle[IDX_BETA_A6],
        config,
        is_defender=False
    )

    # Update the attraction for action i
    new_attraction = (
        (particle[IDX_PHI] * particle[IDX_N] * particle[IDX_A1 + a])
        + (
            particle[IDX_DELTA]
            + (1 - particle[IDX_DELTA])
            * jnp.where(attacker_action == a, 1, 0)
        )
        * attacker_utility
    ) / new_experience
    particle = particle.at[IDX_A1 + a].set(new_attraction)
    return (
        particle,
        attacker_observation,
        attacker_action,
        defender_action,
        config,
        new_experience
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

    #Collect latent state values from the particle
    latent_state = jnp.array([
        particle[IDX_XD1],
        particle[IDX_XD2],
        particle[IDX_XD3],
        particle[IDX_XD4],
        particle[IDX_XA1],
        particle[IDX_XA2],
        particle[IDX_XA3],
        particle[IDX_XA4],
    ])

    #Now we add our estimate of the noise that the attacker experiences

    noise = jax.random.normal(key, shape=(8,)) * particle[IDX_SIGMA]

    attacker_observation = latent_state + noise

    # Now we perform the EWA updates
    new_experience = particle[IDX_RHO] * particle[IDX_N] + 1
    carry = (
        particle,
        attacker_observation,
        attacker_action,
        defender_action,
        config,
        new_experience,
    )
    updated_particle, *_ = jax.lax.fori_loop(
        0, num_actions, single_attraction_update, carry
    )
    updated_particle = updated_particle.at[IDX_N].set(new_experience)
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

def transition_latent_states(key, particles, attacker_action, defender_action, config):

    keys = jax.random.split(key, NUM_PARTICLES)

    particles = jax.vmap(_transition_latent_state, in_axes=(0, 0, None, None, None))(
        keys, particles, attacker_action, defender_action, config
    )

    return particles

transition_latent_states = jax.jit(transition_latent_states)

def _transition_latent_state(key, particle, attacker_action, defender_action, config:MainConfig):

    #Collect latent state values from the particle
    latent_state = jnp.array([
        particle[IDX_XD1],
        particle[IDX_XD2],
        particle[IDX_XD3],
        particle[IDX_XD4],
        particle[IDX_XA1],
        particle[IDX_XA2],
        particle[IDX_XA3],
        particle[IDX_XA4],
    ])

    #Now we add our estimate of the latent state noise

    noise = jax.random.normal(key, shape=(8,)) * config.game.error_X_sigma

    latent_state = latent_state + noise

    defender_inputs = config.game.defender_action_set[defender_action]

    #Ok now we transition to the post-decision state
    XD1_post, XD2_post, XD3_post, XD4_post = kinematic_transition(
        latent_state[0],
        latent_state[1],
        latent_state[2],
        latent_state[3],
        config.game.L,
        config.game.delta_t,
        defender_inputs[0],
        defender_inputs[1]
    )

    attacker_inputs = config.game.attacker_action_set[attacker_action]

    XA1_post, XA2_post, XA3_post, XA4_post = kinematic_transition(
        latent_state[4],
        latent_state[5],
        latent_state[6],
        latent_state[7],
        config.game.L,
        config.game.delta_t,
        attacker_inputs[0],
        attacker_inputs[1]
    )

    #Stack the values
    latent_state = jnp.array([
        XD1_post,
        XD2_post,
        XD3_post,
        XD4_post,
        XA1_post,
        XA2_post,
        XA3_post,
        XA4_post,
    ])
    particle = particle.at[IDX_XD1 : IDX_XA4 + 1].set(
        latent_state
    )  # Insert the transitioned values into the matrix
    return particle

transition_latent_states = jax.jit(transition_latent_states)

update_action_probabilities_all = jax.jit(update_action_probabilities_all)


def get_log_probability_observation(particle, observation, sigmas):
    """
    Compute log P(Y_t | X_t) for an 8D latent state and 8D observation,
    assuming independent Gaussian noise per dimension.

    Args:
        particle: 1D array containing latent state among other fields.
        observation: jnp.array of shape (8,) with observed [XD1, XD2, XD3, XD4, XA1, XA2, XA3, XA4].
        sigmas: jnp.array of shape (8,) with std dev for each observed dimension.

    Returns:
        probability: scalar joint likelihood P(Y_t | X_t) under the noise model.
    """

    # Extract latent state from the particle in the same order as observation
    latent_state = jnp.array([
        particle[IDX_XD1],
        particle[IDX_XD2],
        particle[IDX_XD3],
        particle[IDX_XD4],
        particle[IDX_XA1],
        particle[IDX_XA2],
        particle[IDX_XA3],
        particle[IDX_XA4],
    ])

    error = observation - latent_state

    # Wrap heading errors
    error = error.at[2].set(jnp.arctan2(jnp.sin(error[2]), jnp.cos(error[2])))
    error = error.at[6].set(jnp.arctan2(jnp.sin(error[6]), jnp.cos(error[6])))

    # Per-dimension likelihoods under N(0, sigma_i^2)
    log_likelihoods = logpdf(error, 0.0, sigmas)  # shape (8,)

    # Joint likelihood = product over dimensions
    probability = jnp.sum(log_likelihoods)

    return probability


def get_log_probability_action(particle, attacker_action):
    """
    This function returns the log probability of the attackers observed action given the attractions of the PREVIOUS step.

    Args:
        particle (array): The particle whose attractions/probabilities we are using to calculate the probability of the attacker's action
        attacker_action (int): The attacker's action

    Returns:
        probability (float): The probability of the attacker playing the observed action under this particles previous probability estimates
    """
    # probability = particle[IDX_P1 + attacker_action]
    # return probability

    p = particle[IDX_P1 + attacker_action]

    # Guard against log(0)
    p = jnp.maximum(p, 1e-12)

    return jnp.log(p)


def get_log_joint_probability_particle(particle, attacker_action, defender_observation, sigmas):
    """
    Gets the joint probability of seeing both the attacker's action and the defender's observation under a specfifc particle's parameters.
    This is also equivalent to the raw weight of the particle without normalization.

    Args:
        particle (array): The particle containing the relevent information
        attacker_action (int): The attacker's observed action

    Returns:
        joint_probability (float): The product of the action probability and the observation probability
    """
    # joint_probability = get_probability_action(
    #     particle, attacker_action
    # ) * get_probability_observation(particle, defender_observation, sigmas)
    # return joint_probability

    log_p_action = get_log_probability_action(particle, attacker_action)
    log_p_obs    = get_log_probability_observation(
        particle, defender_observation, sigmas
    )

    return log_p_action + log_p_obs


def get_weights_all(particles, attacker_action, defender_observation, config: MainConfig):
    """
    Gets the normalized weights for all particles, given an observed attacker action.

    Args:
        particles (array): The array of particles, with each entry being a particle
        attacker_action (int): The attacker's observed action

    Returns:
        norm_weights (array): The normalized weights of all particles
    """

    # simgas = config.game.error_Y_sigma

    # raw_weights = jax.vmap(get_log_joint_probability_particle, in_axes=(0, None, None, None))(
    #     particles, attacker_action, defender_observation, simgas
    # )
    # norm_weights = normalize(raw_weights)
    # return norm_weights

    sigmas = config.game.error_Y_sigma

    log_weights = jax.vmap(
        get_log_joint_probability_particle,
        in_axes=(0, None, None, None),
    )(particles, attacker_action, defender_observation, sigmas)

    # Normalize in log-space
    log_weights = log_weights - logsumexp(log_weights)

    # Convert to probabilities ONLY if needed
    weights = jnp.exp(log_weights)

    return weights


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
    weights = get_weights_all(particles, attacker_action, defender_observation, config)

    ESS = effective_sample_size(weights)
    N = particles.shape[0]

    # jax.debug.print(
    #     "ESS = {ess:.1f} / {N}",
    #     ess=ESS,
    #     N=N
    # )


    # Resample particles according to the normalized weights (Step 7)
    key, subkey = random.split(key)
    particles, indices = resample_particles(subkey, particles, weights)

    # Update the attractions and probabilities via EWA update (Steps 9]-11)
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
    Adapts to the number of attacker actions in config.
    Only shows up to `max_show` entries for attractions/probabilities.
    """
    K = int(config.game.num_attacker_actions)  # e.g., 9

    print("-" * 50)

    # --- EWA Core Parameters ---
    print("EWA Parameters:")
    print(f"{'DELTA':<10} (idx={IDX_DELTA}):   {particle[IDX_DELTA]}")
    print(f"{'PHI':<10} (idx={IDX_PHI}):     {particle[IDX_PHI]}")
    print(f"{'RHO':<10} (idx={IDX_RHO}):     {particle[IDX_RHO]}")
    print(f"{'LAMBDA':<10} (idx={IDX_LAMBDA}): {particle[IDX_LAMBDA]}")
    print(f"{'N':<10} (idx={IDX_N}):       {particle[IDX_N]}")

    # --- Beta Parameters ---
    print("\nBeta Parameters (Attacker payoffs):")
    print(f"{'BETA_A1':<10} (idx={IDX_BETA_A1}): {particle[IDX_BETA_A1]}")
    print(f"{'BETA_A2':<10} (idx={IDX_BETA_A2}): {particle[IDX_BETA_A2]}")
    print(f"{'BETA_A3':<10} (idx={IDX_BETA_A3}): {particle[IDX_BETA_A3]}")
    print(f"{'BETA_A4':<10} (idx={IDX_BETA_A4}): {particle[IDX_BETA_A4]}")

    # --- Noise Parameter ---
    print("\nNoise Parameter:")
    print(f"{'SIGMA':<10} (idx={IDX_SIGMA}):   {particle[IDX_SIGMA]}")

    # --- Latent State ---
    print("\nLatent State (Defender):")
    print(f"{'XD1':<10} (idx={IDX_XD1}):     {particle[IDX_XD1]}")  # horiz
    print(f"{'XD2':<10} (idx={IDX_XD2}):     {particle[IDX_XD2]}")  # vert
    print(f"{'XD3':<10} (idx={IDX_XD3}):     {particle[IDX_XD3]}")  # heading
    print(f"{'XD4':<10} (idx={IDX_XD4}):     {particle[IDX_XD4]}")  # speed

    print("\nLatent State (Attacker):")
    print(f"{'XA1':<10} (idx={IDX_XA1}):     {particle[IDX_XA1]}")  # horiz
    print(f"{'XA2':<10} (idx={IDX_XA2}):     {particle[IDX_XA2]}")  # vert
    print(f"{'XA3':<10} (idx={IDX_XA3}):     {particle[IDX_XA3]}")  # heading
    print(f"{'XA4':<10} (idx={IDX_XA4}):     {particle[IDX_XA4]}")  # speed

    # --- Attractions ---
    print("\nAttractions:")
    start_a = IDX_A1
    show_a = min(max_show, K)
    for i in range(show_a):
        idx = start_a + i
        print(f"  A{i+1:<3} (idx={idx}): {particle[idx]}")
    if K > max_show:
        print("  ...")

    # --- Probabilities ---
    print("\nProbabilities:")
    start_p = IDX_P1
    show_p = min(max_show, K)
    for i in range(show_p):
        idx = start_p + i
        print(f"  P{i+1:<3} (idx={idx}): {particle[idx]}")
    if K > max_show:
        print("  ...")

    print("-" * 50)


if __name__ == "__main__":

    key = random.PRNGKey(np.random.randint(0, 100))

    defender_observation = jnp.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    key, subkey = random.split(key)
    config = MainConfig.create(subkey)

    attacker_action = 4
    defender_action = 6

    key, subkey = random.split(key)
    particles = initialize_particles_beta_prior(
        key, config, config.filter.num_particles, config.game.num_attacker_actions
    )
    print_particle_with_labels(get_means(particles), config)

    key, subkey = random.split(key)
    particles = transition_latent_states(subkey, particles, attacker_action, defender_action, config)
    print_particle_with_labels(get_means(particles), config)

    #Calculate normalized weights (Steps 5)
    weights = get_weights_all(particles, attacker_action, defender_observation, config)

    #Resample particles according to the normalized weights (Step 7)
    key, subkey = random.split(key)
    particles, indices = resample_particles(subkey, particles, weights)

    #Update the attractions and probabilities via EWA update (Steps 9]-11)
    key, subkey = random.split(key)
    particles = update_attractions_all(subkey, particles, attacker_action, defender_action, config)
    particles = update_action_probabilities_all(particles, config.game.num_attacker_actions)

    # Evolve the particle parameters (Steps 12 and 13)
    key, subkey = random.split(key)
    particles = evolve_particle_parameters(subkey, particles, config)
    print_particle_with_labels(get_means(particles), config)
