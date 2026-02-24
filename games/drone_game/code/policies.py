import itertools

import filters as jf
import jax
import jax.numpy as jnp
import regressions as rf
from config import MainConfig
from filters import (
    get_means,
    _transition_latent_state,
    update_attractions,
    _evolve_particle_parameters_single
                      )
from jax import random
from utility_functions import defender_utility, attacker_utility
from utils import safe_softmax
from constants import (
    LOGICAL_PARTICLE_DIM,
    MAX_NUM_ATTACKER_ACTIONS,
    NUM_PARTICLES,
    IDX_A1,
    IDX_P1,
    IDX_A,
    IDX_D,
    IDX_T,
    HIDDEN_DIMS
)
import ADP as adp


def simulate_particle(key, particle, defender_action, num_attacker_actions, config):
    """
    Executes a simulated step forward in the game for a single particle.

    Args:
        key (PRNG_key): A key to sample the perturbation error
        particle (array): The single particle containing the necessary data

    Returns:
        particle (array): The particle passed to the function, no update since there is no hidden state
        utility (float): The defender's utility resulting from the simulation results
    """

    # Read a STATIC slice for the full probability block
    probs_full = particle[IDX_P1 : IDX_P1 + MAX_NUM_ATTACKER_ACTIONS]   # static bounds



    # Mask out inactive actions; K can be a tracer — that's fine
    mask = jnp.arange(MAX_NUM_ATTACKER_ACTIONS) < num_attacker_actions
    probs_full = jnp.where(mask, probs_full, 0.0)

    # (Optional) If probs_full might not be perfectly normalized (should be if you used masked softmax),
    # re-normalize safely:
    s = jnp.sum(probs_full)
    probs_full = jnp.where(s > 0, probs_full / s, probs_full)

    key, subkey = random.split(key)
    # IMPORTANT: random.choice requires len(p) == a. So make a == MAX_NUM_ATTACKER_ACTIONS.
    attacker_action = random.choice(key, a=MAX_NUM_ATTACKER_ACTIONS, p=probs_full)

    #Extract the latent state, i.e the previous defender and attacker actions
    prev_atk_act = particle[IDX_A]
    prev_def_act = particle[IDX_D]
    timestep = jnp.int16(particle[IDX_T])

    utility = defender_utility(attacker_action, defender_action, config,
                               timestep=timestep,
                               prev_attacker_action=prev_atk_act,
                               prev_defender_action=prev_def_act
    )

    particle = _transition_latent_state(particle, attacker_action, defender_action)

    key, subkey = random.split(key)
    particle = update_attractions(key, particle, attacker_action, defender_action, config)

    key, subkey = random.split(key)
    particles = _evolve_particle_parameters_single(subkey, particle, config)
    return particle, utility


def simulate_particles(carry, defender_action):
    """
    Executes a single step simulation given an array of particles and a defender action

    Args:
        defender_action (int): The defender's action
        carry (tuple): An array containing key, particles, and the accumulated utility


    Returns:
        key: The PRNG key
        particles (array): The particles
        cum_utility (float): The cumulative utility from the particles

    Note: The simulated particles are not returned as they are fully internal to the policy
    """
    key, particles, utility, config = carry
    num_attacker_actions = config.game.num_attacker_actions
    keys = random.split(key, NUM_PARTICLES)
    sim_particles, utilities = jax.vmap(simulate_particle, in_axes=(0, 0, None, None, None))(
        keys, sim_particles, defender_action, num_attacker_actions, config
    )
    cum_utility = utility + jnp.sum(utilities)
    new_carry = (key, particles, cum_utility, config)
    return new_carry, None

simulate_particles = jax.jit(simulate_particles)

def execute_trajectory(key, particles, trajectory, config: MainConfig):

    """
    Execute the a single trajectory simulation for an array of particles

    Args:
        key: PRNG key
        particles: The particles to simulate with
        trajectory: The sequence of defender actions to use

    Returns:
        average_utility: The average utility received by the particles
            using the given trajectory
    """

    carry_init = (key, particles, 0.0, config)
    carry, _ = jax.lax.scan(simulate_particles, carry_init, trajectory)
    total_utility = carry[-2] #Second to last position in the carry
    average_utility = total_utility / NUM_PARTICLES
    return average_utility


def execute_all_trajectories(key, particles, trajectories, config:MainConfig):

    """
    Execute many trajectories for an array of particles

    Args:
        key: PRNG key
        particles: The particles to simulate with
        trajectores: The array containing the trajectories to use

    Returns:
        utilities of each trajectory in an array
    """

    keys = random.split(key, trajectories.shape[0])
    return jax.vmap(execute_trajectory, in_axes=(0, None, 0, None))(
        keys, particles, trajectories, config
    )


execute_all_trajectories = jax.jit(execute_all_trajectories)


def generate_all_trajectories(H, config:MainConfig):
    """
    A simple function to generate all  ossible action trajectories for the defender across timesteps t+H
    This is for when the policies may make a decision at each epoch.

    Args:
        H (int): The number of additio al steps beyond t. If zero, then trajectories will include one action.

    Returns:
        trajectories (array): An array containing all trajectories of size 1 + H
    """

    tuples = list(itertools.product(jnp.arange(config.game.num_defender_actions), repeat=1 + H))
    trajectories = jnp.array(tuples)
    return trajectories


# def generate_all_trajectories(H, config: MainConfig, repeat: int = 2):
#     """
#     Generate defender action trajectories where each action must be
#     repeated `repeat` times before switching. If the total length
#     (1+H) isn't divisible by `repeat`, the last block is truncated
#     but may be *any* action.

#     This is for the case when the defender makes a decision every other epoch

#     Args:
#         H (int): Number of additional steps beyond t.
#         config (MainConfig): Config object with num_defender_actions.
#         repeat (int): Number of repeats per action before switching.

#     Returns:
#         trajectories (jnp.ndarray): All valid trajectories.
#     """
#     length = 1 + H
#     num_actions = config.game.num_defender_actions

#     # Number of complete repeat blocks
#     full_blocks = length // repeat
#     leftover = length % repeat

#     # Generate all sequences of full blocks
#     block_sequences = itertools.product(range(num_actions), repeat=full_blocks)

#     all_trajs = []
#     for seq in block_sequences:
#         traj = []
#         for a in seq:
#             traj.extend([a] * repeat)
#         if leftover > 0:
#             # Add a leftover block with any action choice
#             for a in range(num_actions):
#                 all_trajs.append(jnp.array(traj + [a] * leftover))
#         else:
#             all_trajs.append(jnp.array(traj))

#     return jnp.stack(all_trajs)



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


def get_H2S_action(key, particles, trajectories, config: MainConfig):
    """
    Returns the action as calculated via the H-horizon simulation policy.
    """

    sampled_utilities = execute_all_trajectories(
        key, particles, trajectories, config
    )
    best_trajectory = get_best_trajectory(sampled_utilities, trajectories)
    H2S_action = best_trajectory[0]
    return H2S_action


get_H2S_action = jax.jit(get_H2S_action)



def generate_cost_function_estimate(particles, defender_action, config:MainConfig):

    num_attacker_actions = config.game.num_attacker_actions

    particle_means = get_means(particles)

    # Read a STATIC slice for the full probability block
    prob_means = particle_means[IDX_P1 : IDX_P1 + MAX_NUM_ATTACKER_ACTIONS]   # static bounds

    # Mask out inactive actions; K can be a tracer — that's fine
    mask = jnp.arange(MAX_NUM_ATTACKER_ACTIONS) < num_attacker_actions
    prob_means = jnp.where(mask, prob_means, 0.0)

    # (Optional) If probs_full might not be perfectly normalized (should be if you used masked softmax),
    # re-normalize safely:
    s = jnp.sum(prob_means)
    prob_means = jnp.where(s > 0, prob_means / s, prob_means)

    #Extract the latent state, i.e the previous defender and attacker actions
    prev_atk_act = particle_means[IDX_A]
    prev_def_act = particle_means[IDX_D]
    timestep = jnp.int16(particle_means[IDX_T])


    def body_func(a, utility):
        util_val = defender_utility(a, defender_action, config,
                               timestep=timestep,
                               prev_attacker_action=prev_atk_act,
                               prev_defender_action=prev_def_act
        )
        scalar_val = jnp.reshape(util_val, ())

        # keep scalar return to match init carry
        return utility + prob_means[a] * scalar_val

    #Loop through all 
    utility_estimate = jax.lax.fori_loop(0, num_attacker_actions, body_func, 0.0)

    return utility_estimate

def get_boltzmann_action(key, particles, defender_actions, config, k=2):

    cost_function_estimates = jax.vmap(
        generate_cost_function_estimate, in_axes=(None, 0, None)
    )(particles, defender_actions, config)
    preferences = safe_softmax(cost_function_estimates, k=k)
    defender_action = random.choice(key, a=defender_actions, p=preferences)
    return defender_action


get_boltzmann_action = jax.jit(get_boltzmann_action, static_argnums=())


def _generate_cost_function_estimate(particle, defender_action, config:MainConfig):
    """
    Alternative _generate_cost_function_estimate for use with a single particle. Used to
    create value estimates for the ADP policy.
    """

    num_attacker_actions = config.game.num_attacker_actions

    # Read a STATIC slice for the full probability block
    probabilities = particle[IDX_P1 : IDX_P1 + MAX_NUM_ATTACKER_ACTIONS]   # static bounds

    # Mask out inactive actions; K can be a tracer — that's fine
    mask = jnp.arange(MAX_NUM_ATTACKER_ACTIONS) < num_attacker_actions
    probabilities = jnp.where(mask, probabilities, 0.0)

    # (Optional) If probs_full might not be perfectly normalized (should be if you used masked softmax),
    # re-normalize safely:
    s = jnp.sum(probabilities)
    probabilities = jnp.where(s > 0, probabilities / s, probabilities)

    #Extract the latent state, i.e the previous defender and attacker actions
    prev_atk_act = particle[IDX_A]
    prev_def_act = particle[IDX_D]
    timestep = jnp.int16(particle[IDX_T])

    def body_func(a, utility):
        util_val = defender_utility(a, defender_action, config,
                               timestep=timestep,
                               prev_attacker_action=prev_atk_act,
                               prev_defender_action=prev_def_act
        )
        scalar_val = jnp.reshape(util_val, ())

        # keep scalar return to match init carry
        return utility + probabilities[a] * scalar_val

    utility_estimate = jax.lax.fori_loop(0, num_attacker_actions, body_func, 0.0)
    return utility_estimate

# def ADP_compute_data(
#     key,
#     config: MainConfig,
#     num_particles,
#     with_model=False,
#     model=jnp.zeros((LOGICAL_PARTICLE_DIM + 1,)),
# ):
#     """
#     This function corresponds to steps 3 - 10 of Algorithm 2

#     Args:

#     Returns:

#     """
#     subkeys = random.split(key, 3)
#     num_attacker_actions = config.game.num_attacker_actions
#     num_defender_actions = config.game.num_defender_actions

#     # Sample the means
#     particle_means = jf.initialize_particles_beta_prior(
#         subkeys[0], config, num_particles, num_attacker_actions
#     )

#     # Sample the decisions
#     defender_actions = random.randint(
#         subkeys[1], shape=(num_particles,), minval=0, maxval=num_defender_actions
#     )

#     # Get more keys
#     subkeys = random.split(subkeys[2], num_particles)

#     # Simulate the particle means forward
#     new_particle_means, _ = jax.vmap(simulate_particle, in_axes=(0, 0, 0))(
#         subkeys, particle_means, defender_actions
#     )

#     # Compute the value estimates for each particle
#     with_model = jnp.asarray(with_model, dtype=bool)
#     value_estimates = jax.vmap(max_value_estimate, in_axes=(0, None, None))(
#         new_particle_means, with_model, model
#     )

#     return particle_means, defender_actions, value_estimates

# def generate_ADP_model(
#     particle_means, defender_actions, value_estimates, model_matrix, t
# ):
#     defender_actions = defender_actions.reshape(-1, 1)
#     design_matrix = rf.combine_features(
#         particle_means[:, :LOGICAL_PARTICLE_DIM], defender_actions
#     )
#     model_coefficients = rf.OLS_regression(design_matrix, value_estimates)
#     model_matrix = model_matrix.at[t].set(model_coefficients)
#     return model_matrix

# def compute_value_functions(key, config: MainConfig, num_timesteps):

#     num_coefficients = LOGICAL_PARTICLE_DIM + 1 # The number of coefficients in each linear regression is the dimensionality of each particle and an extra space for the action

#     # Define matrix to hold regression model parameters, number of rows = number of timesteps as we need a model for each timestep
#     model_matrix = jnp.zeros(shape=(num_timesteps, num_coefficients))
#     num_particles = config.filter.num_particles
#     num_defender_actions = config.game.num_defender_actions
#     num_attacker_actions = config.game.num_attacker_actions

#     final_timestep = num_timesteps - 1  # Adjust for indexing

#     # We first deal with modeling the final timestep, which has V(s, d) = 0 for all states
#     key, subkey = random.split(key)
#     particle_means, defender_actions, value_estimates = ADP_compute_data(
#         subkey, config, num_particles, num_defender_actions
#     )
#     model_matrix = generate_ADP_model(
#         particle_means, defender_actions, value_estimates, model_matrix, final_timestep
#     )

#     # Now that the edge case is handled, we can recursively generate the remaining value functions
#     def _body_function(i, carry):
#         key, config, final_timestep, model_matrix = carry
#         with_model = True
#         model = model_matrix[i]
#         current_timestep = final_timestep - i
#         key, subkey = random.split(key)
#         particle_means, defender_actions, value_estimates = ADP_compute_data(
#             subkey, config, num_particles, num_defender_actions, num_attacker_actions, with_model, model
#         )
#         model_matrix = generate_ADP_model(
#             particle_means,
#             defender_actions,
#             value_estimates,
#             model_matrix,
#             current_timestep,
#         )

#         new_carry = (key, config, final_timestep, model_matrix)
#         return new_carry

#     init_carry = (key, config, final_timestep, model_matrix)
#     *_, model_matrix = jax.lax.fori_loop(1, num_timesteps, _body_function, init_carry)

#     return model_matrix

# def max_value_estimate(particle, with_model, thetas, config:MainConfig):
#     defender_actions = jnp.arange(1, config.game.num_defender_actions+1)
#     value_estimates = jax.lax.cond(
#         with_model,
#         lambda _: jax.vmap(
#             get_value_estimate_with_model, in_axes=(None, 0, None)
#         )(particle, defender_actions, thetas),
#         lambda _: jax.vmap(get_value_estimate_no_model, in_axes=(None, 0))(
#             particle, defender_actions
#         ),
#         operand=None,
#     )
#     return jnp.max(value_estimates)

# def get_value_estimate_no_model(particle, defender_action):
#     """
#     This function calculates the value estimates in the case
#     where t = T and the value of all states after time T is 0
#     """

#     value_estimate = _generate_cost_function_estimate(
#         particle, defender_action
#     )
#     return value_estimate

# def get_value_estimate_with_model(particle, defender_action, thetas):
#     """
#     This function calculates the value estimates in the case
#     where t < T and a regression model for t+1 is available
#     """
#     particle = particle[:LOGICAL_PARTICLE_DIM]
#     X = jnp.concatenate(
#         [jnp.ravel(particle), jnp.ravel(defender_action)]
#     )  # We meed to select the logical subset of the particle
#     yhat = rf.predict_linear(X, thetas)
#     value_estimate = (
#         _generate_cost_function_estimate(particle, defender_action)
#         + yhat.reshape()
#     )
#     return value_estimate

# def get_ADP_action_linear(particle, thetas, config:MainConfig):
#     defender_actions = jnp.arange(1, config.game.num_defender_actions+1)
#     value_estimates = jax.vmap(
#         get_value_estimate_with_model, in_axes=(None, 0, None)
#     )(particle, defender_actions, thetas)
#     best_action = jnp.argmax(value_estimates)
#     return best_action + 1

# get_ADP_action_linear = jax.jit(get_ADP_action_linear)



# def get_ADP_action_MLP(particle, model_params, defender_actions, config:MainConfig):

#     model = rf.MLPRegressor(HIDDEN_DIMS)

#     value_estimates = jax.vmap(
#         adp.get_value_estimate_with_model, in_axes=(None, 0, None, None, None)
#     )(particle, defender_actions, model, model_params, config)
#     best_action = jnp.argmax(value_estimates)
#     return best_action + 1

# if __name__ == "__main__":

#     key = random.key(42)

#     key, subkey = random.split(key)
#     config = MainConfig.create(subkey)
#     num_timesteps = config.game.num_timesteps
#     num_particles = config.filter.num_particles
#     num_defender_actions = config.game.num_defender_actions

#     particle_means, defender_actions, value_estimates = ADP_compute_data(
#         key, config, num_particles, num_defender_actions
#     )
#     print(particle_means.shape)
#     print(defender_actions.shape)
#     print(value_estimates.shape)

#     model_matrix = jnp.zeros(shape=(num_timesteps, 17))

#     model_matrix = generate_ADP_model(
#         particle_means, defender_actions, value_estimates, model_matrix, t=99
#     )
#     print(model_matrix[99])

#     particle_means, defender_actions, value_estimates = ADP_compute_data(
#         key,
#         config,
#         num_particles,
#         num_defender_actions,
#         with_model=True,
#         model=model_matrix[99],
#     )
#     print(particle_means.shape)
#     print(defender_actions.shape)
#     print(value_estimates.shape)
#     model_matrix = generate_ADP_model(
#         particle_means, defender_actions, value_estimates, model_matrix, t=98
#     )

#     print(model_matrix[98])

#     model_matrix = compute_value_functions(key, config)
#     print(model_matrix[0])
#     print(model_matrix[1])
#     print(model_matrix[-1])

#     print(model_matrix.shape)
#     num_zero_rows = jnp.sum(jnp.all(model_matrix == 0, axis=1))
#     print(num_zero_rows)  # Output: 2
