import itertools

import filters as jf
import jax
import jax.numpy as jnp
import regressions as rf
from config import MainConfig
from filters import get_means, update_attractions, update_action_probabilities, _evolve_particle_parameters_single
from jax import random
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

defender_betas = jnp.array([20.0, 50.0, 100.0, 0.1])  # CHANGE ALSO IN CONFIG FILE


def adjusted_sigmoid(x, beta):
    return 1 / (1 + jnp.exp(-(x - beta)))


def simulate_particle(key, particle, defender_action, action_matrix, config:MainConfig):
    """
    Executes a simulated step forward in the game for a single particle.

    Args:
        key (PRNG_key): A key to sample the perturbation error
        particle (array): The single particle containing the necessary data, such as latent state
        defender_action (int): The defender action
        action_matrix (array): An array that represents the change in X from a (attacker, defender) action pair

    Returns:
        new_particle (array): The updated particle including the new latent state
        utility (float): The utility resulting from the simulation results
    """
    subkey1, subkey2, subkey3, subkey4, subkey5 = random.split(key, 5)

    attacker_action = jnp.argmax(
        random.multinomial(subkey1, n=1, p=particle[PROBABILITY_INDICES])
    )
    error_X = random.normal(subkey2)  # Sample the perturbation error for X
    error_Y = random.normal(subkey3) * 3  # Sample the observation error for Y
    transition = particle[IDX_X] + action_matrix[attacker_action, defender_action]
    new_state = transition + error_X
    observation = new_state + error_Y
    utility = get_utility(defender_betas, observation, defender_action)
    new_particle = particle.at[IDX_X].set(new_state)
    new_particle = update_attractions(subkey4, new_particle, attacker_action, defender_action, config)
    new_particle = update_action_probabilities(new_particle)
    new_particle = _evolve_particle_parameters_single(subkey5, new_particle, config)
    return new_particle, utility


def simulate_particles(carry, defender_action):
    """
    Executes a single step simulation given an array of particles and a defender action

    Args:
        defender_action (int): The defender's action
        carry (tuple): An array containing key, particles, action_matrix, and the accumulated utility


    Returns:
        new_particles (array): The updated particles
        average_utility (float): The average utility from the particles
    """

    key, particles, action_matrix, utility, config = carry
    keys = random.split(key, NUM_PARTICLES)
    new_particles, utilities = jax.vmap(simulate_particle, in_axes=(0, 0, None, None, None))(
        keys, particles, defender_action, action_matrix, config
    )
    new_utility = utility + jnp.sum(utilities)
    new_carry = (key, new_particles, action_matrix, new_utility, config)
    return new_carry, None


simulate_particles = jax.jit(simulate_particles)

def execute_trajectory(key, particles, trajectory, action_matrix, config:MainConfig):
    carry_init = (key, particles, action_matrix, 0.0, config)
    carry, _ = jax.lax.scan(simulate_particles, carry_init, trajectory)
    total_utility = carry[-2]
    average_utility = total_utility / NUM_PARTICLES
    return average_utility


def execute_all_trajectories(key, particles, trajectories, action_matrix, config:MainConfig):
    keys = random.split(key, trajectories.shape[0])
    return jax.vmap(execute_trajectory, in_axes=(0, None, 0, None, None))(
        keys, particles, trajectories, action_matrix, config
    )


execute_all_trajectories = jax.jit(execute_all_trajectories)


def generate_all_trajectories(H):
    """
    A simple function to generate all  ossible action trajectories for the defender across timesteps t+H

    Args:
        H (int): The number of additio al steps beyond t. If zero, then trajectories will include one action.

    Returns:
        trajectories (array): An array containing all trajectories of size 1 + H
    """

    tuples = list(itertools.product([0, 1, 2], repeat=1 + H))
    trajectories = jnp.array(tuples)
    return trajectories


def get_best_trajectory(sampled_utilities, trajectories):
    """
    Returns the trajectory that produced the highest utility.

    Args:
        sampled_utilities (array): Array of utility values, one per trajectory
        trajectories (array): Array of shape (num_trajectories, T) containing all trajectories

    Returns:
        best_trajectory (array): The trajectory with the highest utility
    """
    best_index = jnp.argmax(sampled_utilities)
    best_trajectory = trajectories[best_index]
    return best_trajectory


def get_H2S_action(key, particles, trajectories, action_matrix, config:MainConfig):
    """
    Returns the action as calculated via the H-horizon simulation policy.
    """

    sampled_utilities = execute_all_trajectories(
        key, particles, trajectories, action_matrix, config
    )
    best_trajectory = get_best_trajectory(sampled_utilities, trajectories)
    H2S_action = best_trajectory[0]
    return H2S_action


get_H2S_action = jax.jit(get_H2S_action)


# def generate_cost_function_estimate(particles, defender_action, action_matrix):
#     particle_means = get_means(particles)
#     probability_means = particle_means[PROBABILITY_INDICES]
#     X_bar = particle_means[X_INDICES]

#     def body_func(a, utility):
#         Y_hat = X_bar + action_matrix[a, defender_action]
#         utility = utility + probability_means[a] * get_utility(defender_betas, Y_hat, defender_action)
#         return utility[1]


#     utility_estimate = jax.lax.fori_loop(0, NUM_ACTIONS, body_func, 0.0)
#     return utility_estimate


def generate_cost_function_estimate(particles, defender_action, action_matrix):
    particle_means = get_means(particles)
    probability_means = particle_means[PROBABILITY_INDICES]
    X_bar = particle_means[X_INDICES]

    # jax.debug.print("particle_means: {}", particle_means)
    # jax.debug.print("probability_means: {}", probability_means)

    def body_func(a, utility):
        Y_hat = X_bar + action_matrix[a, defender_action]
        util_val = get_utility(defender_betas, Y_hat, defender_action)
        scalar_val = jnp.reshape(util_val, ())

        # jax.debug.print(
        #     "iter {a}: prob={p}, Y_hat={y}, util_val={u}, carry_in={c}",
        #     a=a,
        #     p=probability_means[a],
        #     y=Y_hat,
        #     u=util_val,
        #     c=utility
        # )

        # keep scalar return to match init carry
        return utility + probability_means[a] * scalar_val
        # return utility + probability_means[a] * jnp.mean(util_val)

    utility_estimate = jax.lax.fori_loop(0, NUM_ACTIONS, body_func, 0.0)

    # jax.debug.print("Final utility_estimate: {}", utility_estimate)

    return utility_estimate


def _generate_cost_function_estimate(particle, defender_action, action_matrix):
    """
    Alternative _generate_cost_function_estimate for use with a single particle. Used to
    create value estimates for the ADP policy.
    """

    probabilities = particle[PROBABILITY_INDICES]
    X_bar = particle[X_INDICES]

    def body_func(a, utility):
        Y_hat = X_bar + action_matrix[a, defender_action]
        utility = utility + probabilities[a] * get_utility(
            defender_betas, Y_hat, defender_action
        )
        return utility[1]

    utility_estimate = jax.lax.fori_loop(0, NUM_ACTIONS, body_func, 0.0)
    return utility_estimate


def get_boltzmann_action(key, particles, action_matrix, k=2):
    actions = jnp.array([0, 1, 2])
    cost_function_estimates = jax.vmap(
        generate_cost_function_estimate, in_axes=(None, 0, None)
    )(particles, actions, action_matrix)
    # preferences = jax.scipy.special.softmax(cost_function_estimates) * k
    # jax.debug.print(
    #         "cost_function_esimates: {a}",
    #         a=cost_function_estimates
    #     )
    preferences = safe_softmax(cost_function_estimates, k=k)
    return jnp.argmax(random.multinomial(key, n=1, p=preferences))


get_boltzmann_action = jax.jit(get_boltzmann_action)


def ADP_compute_data(
    key,
    config: MainConfig,
    num_particles,
    num_defender_actions,
    with_model=False,
    model=jnp.zeros((18,)),

):
    """
    This function corresponds to steps 3 - 10 of Algorithm 2

    Args:

    Returns:

    """
    subkeys = random.split(key, 3)
    action_matrix = config.game.action_matrix

    # Sample the means
    particle_means = jf.initialize_particles_beta_prior(
        subkeys[0], config, num_particles, num_defender_actions
    )

    # Sample the decisions
    defender_actions = random.randint(
        subkeys[1], shape=(num_particles,), minval=0, maxval=num_defender_actions
    )

    # Get more keys
    subkeys = random.split(subkeys[2], num_particles)

    # Simulate the particle means forward
    new_particle_means, _ = jax.vmap(simulate_particle, in_axes=(0, 0, 0, None, None))(
        subkeys, particle_means, defender_actions, action_matrix, config
    )

    # Compute the value estimates for each particle
    with_model = jnp.asarray(with_model, dtype=bool)
    value_estimates = jax.vmap(max_value_estimate, in_axes=(0, None, None, None))(
        new_particle_means, action_matrix, with_model, model
    )

    return particle_means, defender_actions, value_estimates


def generate_ADP_model(
    particle_means, defender_actions, value_estimates, model_matrix, t
):
    defender_actions = defender_actions.reshape(-1, 1)
    design_matrix = rf.combine_features(
        particle_means[:, :LOGICAL_PARTICLE_DIM], defender_actions
    )
    model_coefficients = rf.OLS_regression(design_matrix, value_estimates)
    model_matrix = model_matrix.at[t].set(model_coefficients)
    return model_matrix


def compute_value_functions(key, config: MainConfig, num_timesteps):

    num_coefficients = 18  # The number of coefficients in each linear regression is the dimensionality of each particle (17), the decision (1)
    # Define matrix to hold regression model parameters, number of rows = number of timesteps as we need a model for each timestep
    model_matrix = jnp.zeros(shape=(num_timesteps, num_coefficients))
    num_particles = config.filter.num_particles
    num_defender_actions = config.game.num_defender_actions

    final_timestep = num_timesteps - 1  # Adjust for indexing

    # We first deal with modeling the final timestep, which has V(s, d) = 0 for all states
    key, subkey = random.split(key)
    particle_means, defender_actions, value_estimates = ADP_compute_data(
        subkey, config, num_particles, num_defender_actions
    )
    model_matrix = generate_ADP_model(
        particle_means, defender_actions, value_estimates, model_matrix, final_timestep
    )

    # Now that the edge case is handled, we can recursively generate the remaining value functions
    def _body_function(i, carry):
        key, config, final_timestep, model_matrix = carry
        with_model = True
        model = model_matrix[i]
        current_timestep = final_timestep - i
        key, subkey = random.split(key)
        particle_means, defender_actions, value_estimates = ADP_compute_data(
            subkey, config, num_particles, num_defender_actions, with_model, model
        )
        model_matrix = generate_ADP_model(
            particle_means,
            defender_actions,
            value_estimates,
            model_matrix,
            current_timestep,
        )

        new_carry = (key, config, final_timestep, model_matrix)
        return new_carry

    init_carry = (key, config, final_timestep, model_matrix)
    *_, model_matrix = jax.lax.fori_loop(1, num_timesteps, _body_function, init_carry)

    return model_matrix


def max_value_estimate(particle, action_matrix, with_model, thetas):
    actions = jnp.array([0, 1, 2])
    # with_model = jnp.asarray(with_model, dtype=bool)
    value_estimates = jax.lax.cond(
        with_model,
        lambda _: jax.vmap(
            get_value_estimate_with_model, in_axes=(None, 0, None, None)
        )(particle, actions, thetas, action_matrix),
        lambda _: jax.vmap(get_value_estimate_no_model, in_axes=(None, 0, None))(
            particle, actions, action_matrix
        ),
        operand=None,
    )
    return jnp.max(value_estimates)


def get_value_estimate_no_model(particle, defender_action, action_matrix):
    """
    This function calculates the value estimates in the case
    where t = T and the value of all states after time T is 0
    """

    value_estimate = _generate_cost_function_estimate(
        particle, defender_action, action_matrix
    )
    return value_estimate


def get_value_estimate_with_model(particle, defender_action, thetas, action_matrix):
    """
    This function calculates the value estimates in the case
    where t < T and a regression model for t+1 is available
    """
    particle = particle[:LOGICAL_PARTICLE_DIM]
    X = jnp.concatenate(
        [jnp.ravel(particle), jnp.ravel(defender_action)]
    )  # We meed to select the logical subset of the particle
    yhat = rf.predict_linear(X, thetas)
    value_estimate = (
        _generate_cost_function_estimate(particle, defender_action, action_matrix)
        + yhat.reshape()
    )
    return value_estimate


def get_ADP_action_linear(particle, thetas, num_defender_actions: int, action_matrix):

    num_actions = num_defender_actions
    action_matrix = action_matrix

    defender_actions = jnp.arange(num_actions)
    value_estimates = jax.vmap(
        get_value_estimate_with_model, in_axes=(None, 0, None, None)
    )(particle, defender_actions, thetas, action_matrix)
    # jax.debug.print("particle: {}", particle)
    # jax.debug.print("thetas: {}", thetas)
    # jax.debug.print("value_estimates: {}", value_estimates)
    best_action = jnp.argmax(value_estimates)
    return best_action


get_ADP_action_linear = jax.jit(get_ADP_action_linear, static_argnums=(2,))

if __name__ == "__main__":

    key = random.key(42)

    key, subkey = random.split(key)
    config = MainConfig.create(subkey)
    num_timesteps = config.game.num_timesteps
    num_particles = config.filter.num_particles
    num_defender_actions = config.game.num_defender_actions

    particle_means, defender_actions, value_estimates = ADP_compute_data(
        key, config, num_particles, num_defender_actions
    )
    print(particle_means.shape)
    print(defender_actions.shape)
    print(value_estimates.shape)

    model_matrix = jnp.zeros(shape=(num_timesteps, 17))

    model_matrix = generate_ADP_model(
        particle_means, defender_actions, value_estimates, model_matrix, t=99
    )
    print(model_matrix[99])

    particle_means, defender_actions, value_estimates = ADP_compute_data(
        key,
        config,
        num_particles,
        num_defender_actions,
        with_model=True,
        model=model_matrix[99],
    )
    print(particle_means.shape)
    print(defender_actions.shape)
    print(value_estimates.shape)
    model_matrix = generate_ADP_model(
        particle_means, defender_actions, value_estimates, model_matrix, t=98
    )

    print(model_matrix[98])

    model_matrix = compute_value_functions(key, config)
    print(model_matrix[0])
    print(model_matrix[1])
    print(model_matrix[-1])

    print(model_matrix.shape)
    num_zero_rows = jnp.sum(jnp.all(model_matrix == 0, axis=1))
    print(num_zero_rows)  # Output: 2
