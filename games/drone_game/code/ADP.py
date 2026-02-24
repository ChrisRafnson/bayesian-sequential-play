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


MODEL = rf.MLPRegressor(hidden_dims=HIDDEN_DIMS)

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
    particle = _evolve_particle_parameters_single(subkey, particle, config)
    return particle, utility

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
        utility = defender_utility(a, defender_action, config,
                            timestep=timestep,
                            prev_attacker_action=prev_atk_act,
                            prev_defender_action=prev_def_act
        )
        scalar_val = jnp.reshape(utility, ())

        # keep scalar return to match init carry
        return utility + probabilities[a] * scalar_val

    utility_estimate = jax.lax.fori_loop(0, num_attacker_actions, body_func, 0.0)
    return utility_estimate

def ADP_compute_data(
    key,
    config: MainConfig,
    num_particles,
    model,
    model_params,
    timestep,
    with_model=False,
):
    """
    This function corresponds to steps 3 - 10 of Algorithm 2

    Args:

    Returns:

    """
    subkeys = random.split(key, 4)
    num_attacker_actions = config.game.num_attacker_actions
    num_defender_actions = config.game.num_defender_actions

    # Sample the means
    particle_means = jf.initialize_particles_beta_prior(
        subkeys[0], config, num_particles, num_attacker_actions, timestep=timestep
    )

    # Sample the  defender actions
    defender_actions = random.randint(
        subkeys[1], shape=(num_particles,), minval=0, maxval=num_defender_actions
    )

    #If the timestep is an odd number, then we need to also sample the "previous" attacker actions
    if timestep % 2 == 1:

        prev_attacker_actions = random.randint(
        subkeys[2], shape=(num_particles,), minval=0, maxval=num_attacker_actions
        )
        
        #The defender is also restricted to repeating its last action on an odd-numebered timestep
        #Therefore his previous actions are identical to the sampled actions for this timestep

        prev_defender_actions = defender_actions

        #We now insert these samples into the particles
        particle_means = particle_means.at[:, IDX_A].set(prev_attacker_actions)
        particle_means = particle_means.at[:, IDX_D].set(prev_defender_actions)


    # Get more keys
    subkeys = random.split(subkeys[3], num_particles)

    # Simulate the particle means forward
    new_particle_means, _ = jax.vmap(simulate_particle, in_axes=(0, 0, 0, None, None))(
        subkeys, particle_means, defender_actions, num_attacker_actions, config
    )

    # Compute the value estimates for each particle
    with_model = jnp.asarray(with_model, dtype=bool)
    value_estimates = jax.vmap(max_value_estimate, in_axes=(0, None, None, None, None))(
        new_particle_means, with_model, model, model_params, config
    )

    return particle_means, defender_actions, value_estimates

def generate_ADP_model(
    particle_means, defender_actions, value_estimates, model_sequence, t, num_epochs: int = 100
):
    """
    Trains the t-th MLP regressor for several epochs.
    """
    defender_actions = defender_actions.reshape(-1, 1)
    design_matrix = rf.combine_features(particle_means, defender_actions)

    # Run multiple epochs
    for _ in range(num_epochs):
        model_sequence[t] = rf.train_step(model_sequence[t], design_matrix, value_estimates)

    return model_sequence

def generate_ADP_model_advanced(
    particle_means, defender_actions, value_estimates,
    model_sequence, t, num_epochs: int = 100,
    patience: int = 10, validation_split: float = 0.2
):
    """
    Train the t-th MLP regressor using Adam with early stopping and rollback.
    """
    # Split data into train / validation
    defender_actions = defender_actions.reshape(-1, 1)
    X = rf.combine_features(particle_means, defender_actions)
    y = value_estimates.reshape(-1, 1)

    n = X.shape[0]
    split_idx = int(n * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    state = model_sequence[t]
    best_state = state
    best_val_loss = jnp.inf
    no_improve = 0

    for epoch in range(num_epochs):
        # One Adam step
        state = rf.train_step(state, X_train, y_train)

        # Compute validation loss (not jitted)
        preds = state.apply_fn(state.params, X_val)
        val_loss = jnp.mean((preds - y_val) ** 2)

        print(f"[Model: {t:<10}] | Epoch: {epoch:03d} | Val Loss: {val_loss:.4f}")

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = state
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}, best val loss = {best_val_loss:.5f}")
            break

    model_sequence[t] = best_state
    return model_sequence

def compute_value_functions(key, config: MainConfig, num_timesteps, num_epochs=100):

    input_dimensions = LOGICAL_PARTICLE_DIM + 1 # The number of inputs to the regressor is the dimensionality of the particle + 1 for the defender action

    #Extract simulation parameters
    num_particles = config.filter.num_particles
    num_defender_actions = config.game.num_defender_actions
    num_attacker_actions = config.game.num_attacker_actions

    final_timestep = num_timesteps - 1  # Adjust for indexing

    # Define a sequence to hold the MLP parameters for each timestep
    key, subkey = random.split(key)
    model_seq = rf.initialize_mlp_sequence(num_stages=num_timesteps, input_dim=input_dimensions, hidden_dims=HIDDEN_DIMS, rng=subkey)

    #Create a model to use to apply the model parameters
    model = rf.MLPRegressor(hidden_dims=HIDDEN_DIMS)

    # We first deal with modeling the final timestep, which has V(s, d) = 0 for all states

    #Generate the training data and value estimates
    key, subkey = random.split(key)
    particle_means, defender_actions, value_estimates = ADP_compute_data(
        subkey, config, num_particles, model, model_seq[final_timestep], final_timestep
    )

    model_seq = generate_ADP_model_advanced(
        particle_means, defender_actions, value_estimates, model_seq, final_timestep, num_epochs=num_epochs
    )


    # Now that the edge case is handled, we can recursively generate the remaining value functions
    def _body_function(i, carry):
        key, config, final_timestep, model_seq = carry
        with_model = True
        current_model_params = model_seq[i]
        current_timestep = final_timestep - i

        key, subkey = random.split(key)
        particle_means, defender_actions, value_estimates = ADP_compute_data(
            subkey, config, num_particles, model, current_model_params, current_timestep,  with_model
        )
        model_seq = generate_ADP_model_advanced(
            particle_means,
            defender_actions,
            value_estimates,
            model_seq,
            current_timestep,
            num_epochs=num_epochs
        )

        new_carry = (key, config, final_timestep, model_seq)
        return new_carry

    # init_carry = (key, config, final_timestep, model_seq)
    # *_, model_seq = jax.lax.fori_loop(1, num_timesteps, _body_function, init_carry)


    # ----- run recursion in plain Python -----
    carry = (key, config, final_timestep, model_seq)

    for i in range(1, num_timesteps):
        print(f"Training model: {final_timestep - i}")
        carry = _body_function(i, carry)

    _, _, _, model_seq = carry

    return model_seq

def max_value_estimate(particle, with_model, model, model_params, config:MainConfig):

    """
    Gets the 
    """

    particle = particle[:LOGICAL_PARTICLE_DIM]

    defender_actions = jnp.arange(1, config.game.num_defender_actions+1)
    value_estimates = jax.lax.cond(
        with_model,
        lambda _: jax.vmap(
            get_value_estimate_with_model, in_axes=(None, 0, None, None, None)
        )(particle, defender_actions, model, model_params.params, config), #Have to extract the params of the model explicitly
        lambda _: jax.vmap(get_value_estimate_no_model, in_axes=(None, 0, None))(
            particle, defender_actions, config
        ),
        operand=None,
    )
    return jnp.max(value_estimates)


def get_value_estimate_no_model(particle, defender_action, config:MainConfig):
    """
    This function calculates the value estimates in the case
    where t = T and the value of all states after time T is 0
    """

    value_estimate = _generate_cost_function_estimate(
        particle, defender_action, config)
    return value_estimate

def get_value_estimate_with_model(particle, defender_action, model, model_params, config:MainConfig):
    """
    This function calculates the value estimates in the case
    where t < T and a regression model for t+1 is available
    """
    X = jnp.concatenate(
        [jnp.ravel(particle), jnp.ravel(defender_action)]
    )


    #Now that the inputs are prepared, we can regress on them
    yhat = model.apply(model_params, X[None, :]).squeeze()
    value_estimate = (
        _generate_cost_function_estimate(particle, defender_action, config)
        + yhat.reshape()
    )
    return value_estimate

def get_ADP_action_MLP(particle, model_params, defender_actions, config:MainConfig):

    model = rf.MLPRegressor(HIDDEN_DIMS)

    value_estimates = jax.vmap(
        get_value_estimate_with_model, in_axes=(None, 0, None, None, None)
    )(particle, defender_actions, model, model_params, config)
    best_action = jnp.argmax(value_estimates)
    return best_action

if __name__ == "__main__":

    key = jax.random.PRNGKey(42)
    key, subkey = random.split(key)
    config = MainConfig.create(subkey)

    model_seq = compute_value_functions(key, config, config.game.num_timesteps)