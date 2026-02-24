"""
Approximate Dynamic Programming (ADP) utilities for the Drone Game.

This module:
- Generates training data by simulating particles forward.
- Fits a separate value-function approximator for each timestep (backward recursion).
- Evaluates ADP value estimates by maximizing over defender actions.

Notes:
- All JAX-based simulation / cost estimation is JIT/vmap-friendly.
- Regression here uses a Flax/JAX MLP (see regressions.py), so the full ADP
  pipeline can remain within JAX. If you swap in sklearn models, keep them
  *outside* any jitted function.
"""

from __future__ import annotations

from typing import Tuple

import jax
import gc
import jax.numpy as jnp
from jax import random
from tqdm import trange

import filters as jf
import regressions as rf
from config_ADP_prior import MainConfig
from filters import (
    _transition_latent_state,
    update_attractions,
    _evolve_particle_parameters_single,
    print_particle_with_labels,
    update_action_probabilities_all,
    _update_action_probabilities_masked
)
from utility_functions import utility
from utils import kinematic_transition
from constants import (
    LOGICAL_PARTICLE_DIM,
    MAX_NUM_ATTACKER_ACTIONS,
    HIDDEN_DIMS,
    IDX_P1,
    IDX_P9,
    IDX_XD1,
    IDX_XD2,
    IDX_XD3,
    IDX_XD4,
    IDX_XA1,
    IDX_XA2,
    IDX_XA3,
    IDX_XA4,
)
from policies import get_means
from regressions import huber_loss



# -----------------------------------------------------------------------------
# Core simulation helpers
# -----------------------------------------------------------------------------

def normalize(x, low, high, eps=1e-8):
    """Map x from [low, high] → [0, 1] with safety."""
    u = (x - low) / (high - low + eps)
    return jnp.clip(u, 0.0, 1.0)

def denormalize(u, low, high):
    """Map u from [0, 1] → [low, high]."""
    return low + u * (high - low)

def centered_log_probs(probs, eps=1e-12):
    probs = jnp.clip(probs, eps, 1.0)
    logp = jnp.log(probs)
    return logp - jnp.mean(logp)



def _one_hot_def_actions(defender_actions: jnp.ndarray, num_defender_actions: int) -> jnp.ndarray:
    """
    defender_actions:
      - (N,) int array, or
      - scalar int
    returns:
      - (N, K) if input is (N,)
      - (K,) if input is scalar
    """
    defender_actions = jnp.asarray(defender_actions, dtype=jnp.int32)
    return jax.nn.one_hot(defender_actions, num_defender_actions)


def _masked_action_probs(
    probs_full: jnp.ndarray,
    num_attacker_actions: int,
) -> jnp.ndarray:
    """
    Zero out probabilities for inactive actions and (safely) renormalize.

    Args:
        probs_full: length MAX_NUM_ATTACKER_ACTIONS probability vector.
        num_attacker_actions: number of *active* attacker actions (<= MAX).

    Returns:
        probs: length MAX_NUM_ATTACKER_ACTIONS, masked and normalized.
    """
    mask = jnp.arange(MAX_NUM_ATTACKER_ACTIONS) < num_attacker_actions
    probs = jnp.where(mask, probs_full, 0.0)
    s = jnp.sum(probs)
    return jnp.where(s > 0, probs / s, probs)


def simulate_particle(
    key: jax.Array,
    particle: jnp.ndarray,
    defender_action: int,
    num_attacker_actions: int,
    config: MainConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate a single particle forward by one step.

    Returns:
        new_particle: updated particle after transition + learning updates.
        def_utility: defender utility for the simulated transition.
    """
    # --- sample attacker action from particle belief ---
    probs_full = particle[IDX_P1 : IDX_P1 + MAX_NUM_ATTACKER_ACTIONS]
    probs = _masked_action_probs(probs_full, num_attacker_actions)
    

    key, subkey = random.split(key)
    attacker_action = random.choice(subkey, a=MAX_NUM_ATTACKER_ACTIONS, p=probs)

    # --- transition latent state (dynamics) ---
    key, subkey = random.split(key)
    particle = _transition_latent_state(subkey, particle, attacker_action, defender_action, config)

    # --- compute utility from (post-transition) state ---
    def_utility = utility(
        particle[IDX_XD2],   # "true" vertical (if your utility expects it)
        particle[IDX_XD1],
        particle[IDX_XD2],
        particle[IDX_XD3],
        particle[IDX_XD4],
        particle[IDX_XA1],
        particle[IDX_XA2],
        particle[IDX_XA3],
        particle[IDX_XA4],
        defender_action,
        config.game.defender_beta_1,
        config.game.defender_beta_2,
        config.game.defender_beta_3,
        config.game.defender_beta_4,
        config.game.defender_beta_5,
        config.game.defender_beta_6,
        config,
        is_defender=True
    )

    # --- learning updates on the particle ---
    key, subkey = random.split(key)
    particle = update_attractions(subkey, particle, attacker_action, defender_action, config)

    K = jnp.int16(config.game.num_attacker_actions)
    particle = _update_action_probabilities_masked(particle, K)

    key, subkey = random.split(key)
    particle = _evolve_particle_parameters_single(subkey, particle, config)

    return particle, def_utility


# -----------------------------------------------------------------------------
# Cost-to-go estimate (JAX-friendly)
# -----------------------------------------------------------------------------

def generate_cost_function_estimate(
    particle: jnp.ndarray,
    defender_action: int,
    config: MainConfig,
) -> jnp.ndarray:
    """
    JAX-friendly estimate of immediate expected utility for a fixed defender action.

    This computes:
        E_a[ U(s' | a, d) ]
    where the expectation is under the particle's attacker mixed strategy,
    and s' is the *post-kinematic* state (using kinematic_transition).

    Returns:
        scalar utility estimate (shape ()) suitable for vmap / jit.
    """
    num_attacker_actions = config.game.num_attacker_actions

    # Mask & normalize attacker action probabilities
    probs_full = particle[IDX_P1 : IDX_P1 + MAX_NUM_ATTACKER_ACTIONS]
    probabilities = _masked_action_probs(probs_full, num_attacker_actions)

    # Extract latent state (pre-decision)
    XD1 = particle[IDX_XD1]
    XD2 = particle[IDX_XD2]
    XD3 = particle[IDX_XD3]
    XD4 = particle[IDX_XD4]
    XA1 = particle[IDX_XA1]
    XA2 = particle[IDX_XA2]
    XA3 = particle[IDX_XA3]
    XA4 = particle[IDX_XA4]

    L = config.game.L
    dt = config.game.delta_t

    # Defender action (accel, steer)
    def_inputs = config.game.defender_action_set[defender_action]

    def body_func(a: int, cum: jnp.ndarray) -> jnp.ndarray:
        """Accumulate probabilities[a] * utility for attacker action a."""
        atk_inputs = config.game.attacker_action_set[a]

        # Kinematic step for defender
        XD1p, XD2p, XD3p, XD4p = kinematic_transition(
            XD1, XD2, XD3, XD4, L, dt, def_inputs[0], def_inputs[1]
        )

        # Kinematic step for attacker
        XA1p, XA2p, XA3p, XA4p = kinematic_transition(
            XA1, XA2, XA3, XA4, L, dt, atk_inputs[0], atk_inputs[1]
        )

        def_util = utility(
            XD2p,
            XD1p,
            XD2p,
            XD3p,
            XD4p,
            XA1p,
            XA2p,
            XA3p,
            XA4p,
            defender_action,
            config.game.defender_beta_1,
            config.game.defender_beta_2,
            config.game.defender_beta_3,
            config.game.defender_beta_4,
            config.game.defender_beta_5,
            config.game.defender_beta_6,
            config,
            is_defender=True
        )
        def_util = jnp.reshape(def_util, ())
        return cum + probabilities[a] * def_util

    return jax.lax.fori_loop(0, num_attacker_actions, body_func, 0.0)


# -----------------------------------------------------------------------------
# Training data generation + model fitting
# -----------------------------------------------------------------------------

def adp_compute_data(
    key: jax.Array,
    config: MainConfig,
    num_particles: int,
    with_model: bool,
    model_state_next,
    timestep
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate (X, action, y) samples for ADP training at a given recursion step.

    Steps:
      1) sample particles from prior
      2) sample random defender actions
      3) simulate one step forward for each particle (to update particle params)
      4) compute max_a [ cost(s,a) + V_{t+1}(s,a) ] for each updated particle

    Args:
        with_model: if True, include the learned value approximation for t+1.
        model_state_next: the trained model state for t+1 (or final step model).

    Returns:
        particle_means: (N, particle_dim)
        defender_actions: (N,)
        value_estimates: (N,)
    """
    num_attacker_actions = config.game.num_attacker_actions
    num_defender_actions = config.game.num_defender_actions

    key, k_particles, k_actions, k_sim = random.split(key, 4)

    particle_means = jf.initialize_particles_beta_prior(
        k_particles, config, num_particles, num_attacker_actions
    )

    # # -------------------------------
    # # Initial particle diagnostics
    # # -------------------------------
    # jax.debug.print(
    #     "[DEBUG] particle_means shape: {s}",
    #     s=particle_means.shape,
    # )

    # Convert Position Variables to desired range for this timestep

    # ================= Prior bounds from config =================
    XD1_low, XD1_high = config.priors.XD1_bounds
    XD2_low, XD2_high = config.priors.XD2_bounds

    XA1_low, XA1_high = config.priors.XA1_bounds
    XA2_low, XA2_high = config.priors.XA2_bounds

    # ================= Normalize =================
    u_XD1 = normalize(particle_means[:, IDX_XD1], XD1_low, XD1_high)
    u_XD2 = normalize(particle_means[:, IDX_XD2], XD2_low, XD2_high)

    u_XA1 = normalize(particle_means[:, IDX_XA1], XA1_low, XA1_high)
    u_XA2 = normalize(particle_means[:, IDX_XA2], XA2_low, XA2_high)

    dt = config.game.delta_t
    S1, S2 = config.game.S1, config.game.S2
    heading = jnp.deg2rad(40)
    t = timestep

    def horizontal_bounds(x_init, t):
        width = S2 * jnp.sin(heading) * dt * (t + 1)
        return x_init - width, x_init + width

    XD1_low, XD1_high = horizontal_bounds(config.game.XD1_init, t)
    XA1_low, XA1_high = horizontal_bounds(config.game.XA1_init, t)

    # Optional manual horizontal override
    XA1_low, XA1_high = -10.0, 18.0
    XD1_low, XD1_high = -10.0, 18.0


    # ================= Vertical Positions =================
    X2_low = jnp.where(
        t == 0,
        0.0,
        S1 * dt * t
    )

    X2_high = S2 * dt * (t + 1)

    # -------- Vertical override (DROP-IN) --------
    # X2_low, X2_high = 0.0, 120.0   # set to desired bounds

    # ================= Denormalize =================
    XD1_new = denormalize(u_XD1, XD1_low, XD1_high)
    XD2_new = denormalize(u_XD2, X2_low, X2_high)

    XA1_new = denormalize(u_XA1, XA1_low, XA1_high)
    XA2_new = denormalize(u_XA2, X2_low, X2_high)



    # # -------------------------------
    # # Transformed values diagnostics
    # # -------------------------------
    # jax.debug.print(
    #     "[DEBUG] XD1_new: shape={s}, min={mn}, max={mx}",
    #     s=XD1_new.shape,
    #     mn=jnp.min(XD1_new),
    #     mx=jnp.max(XD1_new),
    # )

    # jax.debug.print(
    #     "[DEBUG] XD2_new: shape={s}, min={mn}, max={mx}",
    #     s=XD2_new.shape,
    #     mn=jnp.min(XD2_new),
    #     mx=jnp.max(XD2_new),
    # )

    # jax.debug.print(
    #     "[DEBUG] XA1_new: shape={s}, min={mn}, max={mx}",
    #     s=XA1_new.shape,
    #     mn=jnp.min(XA1_new),
    #     mx=jnp.max(XA1_new),
    # )

    # jax.debug.print(
    #     "[DEBUG] XA2_new: shape={s}, min={mn}, max={mx}",
    #     s=XA2_new.shape,
    #     mn=jnp.min(XA2_new),
    #     mx=jnp.max(XA2_new),
    # )

    # -------------------------------
    # Insert new particles
    # -------------------------------
    particle_means = particle_means.at[:, IDX_XD1].set(XD1_new)
    particle_means = particle_means.at[:, IDX_XD2].set(XD2_new)
    particle_means = particle_means.at[:, IDX_XA1].set(XA1_new)
    particle_means = particle_means.at[:, IDX_XA2].set(XA2_new)

    # # -------------------------------
    # # Post-insertion diagnostics
    # # -------------------------------
    # jax.debug.print(
    #     "[DEBUG] particle_means[IDX_XD1]: shape={s}, min={mn}, max={mx}",
    #     s=particle_means[:, IDX_XD1].shape,
    #     mn=jnp.min(particle_means[:, IDX_XD1]),
    #     mx=jnp.max(particle_means[:, IDX_XD1]),
    # )

    # jax.debug.print(
    #     "[DEBUG] particle_means[IDX_XD2]: shape={s}, min={mn}, max={mx}",
    #     s=particle_means[:, IDX_XD2].shape,
    #     mn=jnp.min(particle_means[:, IDX_XD2]),
    #     mx=jnp.max(particle_means[:, IDX_XD2]),
    # )

    

    defender_actions = random.randint(
        k_actions, shape=(num_particles,), minval=0, maxval=num_defender_actions
    )

    sim_keys = random.split(k_sim, num_particles)

    new_particles, _ = jax.vmap(
        simulate_particle, in_axes=(0, 0, 0, None, None)
    )(sim_keys, particle_means, defender_actions, num_attacker_actions, config)

    # Compute max over defender actions for each particle (JAX MLP model for next step)
    value_estimates = jax.vmap(
        max_value_estimate,
        in_axes=(0, None, None, None),
    )(new_particles, with_model, model_state_next, config)

    # -------------------------------
    # 99% symmetric clipping (CRITICAL)
    # -------------------------------
    abs_vals = jnp.abs(value_estimates)
    vmax = jnp.percentile(abs_vals, 99.0)

    # safety floor so vmax never collapses to 0
    vmax = jnp.maximum(vmax, 1e-3)

    value_estimates = jnp.clip(value_estimates, -vmax, vmax)

    return particle_means, defender_actions, value_estimates




def fit_adp_model_for_timestep(
    particle_means: jnp.ndarray,
    defender_actions: jnp.ndarray,
    value_estimates: jnp.ndarray,
    model_seq,
    config:MainConfig,
    t: int,
    *,
    num_epochs: int = 100,
    patience: int = 10,
    validation_split: float = 0.05,
):
    """
    Fit model_seq[t] using Adam + early stopping (with rollback of best state).

    Change: one-hot encode defender actions and normalize only particle features.
    """
    num_defender_actions = model_seq[t].X_mean.shape[-1] - LOGICAL_PARTICLE_DIM \
        if hasattr(model_seq[t], "X_mean") and model_seq[t].X_mean is not None \
        else None

    # Prefer config-derived K if available; fallback to model state inference above is fragile,
    # so we just use config everywhere else in this file (recommended).
    # We'll compute K directly here using the particle/value inputs' context is not available,
    # so assume you pass it or have it in scope via config in your pipeline.
    # ---- simplest: infer from value_estimates pipeline usage:
    # You already have config in compute_value_functions; so pass K through if you want.
    #
    # For now, we will just use the runtime value from the global model architecture:
    # (See compute_value_functions change below; that sets input_dim = LOGICAL_PARTICLE_DIM + K)
    if num_defender_actions is None:
        raise ValueError(
            "Could not infer num_defender_actions from model state. "
            "Recommended: use config.game.num_defender_actions and build the model with that input_dim."
        )

    # # --- Build features ---
    # particle_means = particle_means[:, :LOGICAL_PARTICLE_DIM]
    # a_oh = _one_hot_def_actions(defender_actions.reshape(-1), num_defender_actions)  # (N, K)
    # X = jnp.concatenate([particle_means, a_oh], axis=1)                              # (N, D+K)
    # y = value_estimates.reshape(-1, 1)

    # --- Build features (using the continuous action features) ---
    particle_means = particle_means[:, :LOGICAL_PARTICLE_DIM]

    # --- replace probability block with centered log-probs ---
    probs = particle_means[:, IDX_P1:IDX_P9 + 1]
    log_probs = centered_log_probs(probs)

    particle_means = particle_means.at[:, IDX_P1:IDX_P9 + 1].set(log_probs)

    # Extract raw actions
    a_cont = config.game.defender_action_set[
        defender_actions.reshape(-1)
    ]  # (N, 2)

    # Normalization constants (define once in config if possible)
    ACCEL_MAX = config.game.max_acceleration    # max |accel|
    STEER_MAX = config.game.max_steering    # max |steering| (degrees)

    a_norm = jnp.stack(
        [
            a_cont[:, 0] / ACCEL_MAX,
            a_cont[:, 1] / STEER_MAX,
        ],
        axis=1
    )

    # Concatenate state + action
    X = jnp.concatenate(
        [particle_means, a_norm],
        axis=1
    )  # (N, D + 2)

    y = value_estimates.reshape(-1, 1)



    # --- Train/val split ---
    n = X.shape[0]
    split_idx = int(n * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # --- Normalize inputs: ONLY particle columns ---
    D = LOGICAL_PARTICLE_DIM
    X_train_p = X_train[:, :D]
    X_val_p   = X_val[:, :D]

    # X_mean_p = jnp.mean(X_train_p, axis=0, keepdims=True)
    # X_std_p  = jnp.std(X_train_p, axis=0, keepdims=True) + 1e-6

    # X_train_pn = (X_train_p - X_mean_p) / X_std_p
    # X_val_pn   = (X_val_p   - X_mean_p) / X_std_p

    eps = 1e-6
    X_mean_p = jnp.mean(X_train_p, axis=0, keepdims=True)
    X_std_p  = jnp.std(X_train_p, axis=0, keepdims=True)

    mask = X_std_p < 1e-2
    X_std_p  = jnp.where(mask, 1.0, X_std_p)          # safe denom
    X_mean_p = jnp.where(mask, 0.0, X_mean_p)         # remove constant-feature bias

    X_train_pn = (X_train_p - X_mean_p) / (X_std_p + eps)
    X_val_pn   = (X_val_p   - X_mean_p) / (X_std_p + eps)

    raw_std_p = jnp.std(X_train_p, axis=0, keepdims=True) + 1e-6

    print(f"[t={t:02d}] raw min std_p: {float(jnp.min(raw_std_p)):.6g}")
    print(f"[t={t:02d}] raw #std_p<1e-2: {int(jnp.sum(raw_std_p < 1e-2))}")

    print(f"[t={t:02d}] USED min std_p: {float(jnp.min(X_std_p)):.6g}")
    print(f"[t={t:02d}] USED #std_p<1e-2: {int(jnp.sum(X_std_p < 1e-2))}")



    # Action one-hot columns are left as-is (0/1)
    X_train = jnp.concatenate([X_train_pn, X_train[:, D:]], axis=1)
    X_val   = jnp.concatenate([X_val_pn,   X_val[:,   D:]], axis=1)

    # Store full-dim mean/std so inference can do (X - mean)/std safely:
    # - particle columns: learned mean/std
    # - one-hot columns: mean=0, std=1 (no-op)
    K = X.shape[1] - D
    X_mean = jnp.concatenate([X_mean_p, jnp.zeros((1, K))], axis=1)
    X_std  = jnp.concatenate([X_std_p,  jnp.ones((1, K))],  axis=1)
    X_std = X_std.at[X_std < 1e-2].set(1.0)


    # --- Normalize targets ---
    y_mean = jnp.mean(y_train, axis=0, keepdims=True)
    y_std  = jnp.std(y_train, axis=0, keepdims=True) + 1e-6
    y_std = jnp.maximum(y_std, 1e-2)


    y_train = (y_train - y_mean) / y_std
    y_val   = (y_val   - y_mean) / y_std

    state = model_seq[t]
    best_state = state
    best_val = jnp.inf
    no_improve = 0

    print(f"[t={t:02d}] X.shape={X.shape}, y.shape={y.shape}")
    print(f"[t={t:02d}] y.mean={float(jnp.mean(y)):.4f}, y.std={float(jnp.std(y)):.4f}")
    print(f"[t={t:02d}] X particle feature means: {jnp.round(jnp.mean(X[:, :D], axis=0), 3)}")
    print(f"[t={t:02d}] X particle feature stds: {jnp.round(jnp.std(X[:, :D], axis=0), 3)}")


    if jnp.std(y_train) < 1e-6:
        print(f"[WARNING] t={t:02d} — y_train nearly constant")

    if jnp.any(jnp.std(X_train, axis=0) < 1e-6):
        print(f"[WARNING] t={t:02d} — degenerate particle feature(s)")



    if jnp.any(jnp.isnan(X_train)) or jnp.any(jnp.isnan(y_train)):
        raise ValueError("NaNs in training data")
    if jnp.any(jnp.isinf(X_train)) or jnp.any(jnp.isinf(y_train)):
        raise ValueError("Infs in training data")


    # for epoch in range(num_epochs):
    #     state = rf.train_step(state, X_train, y_train)

    #     preds = state.apply_fn(state.params, X_val)
    #     val_loss = jnp.mean((preds - y_val) ** 2)

    batch_size = 512
    num_train = X_train.shape[0]

    perm_key = jax.random.PRNGKey(0)

    for epoch in range(num_epochs):
        perm_key, subkey = jax.random.split(perm_key)
        perm = jax.random.permutation(subkey, num_train)

        for i in range(0, num_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            state = rf.train_step(state, xb, yb)

        # validation (unchanged)

        # def batched_apply(params, X, bs=512):
        #     outs = []
        #     for i in range(0, X.shape[0], bs):
        #         outs.append(state.apply_fn(params, X[i:i+bs]))
        #     return jnp.concatenate(outs, axis=0)

        # preds = batched_apply(state.params, X_val, bs=256)

        preds = state.apply_fn(state.params, X_val)
        val_loss = huber_loss(preds, y_val)

        print(f"[ADP model t={t:02d}] epoch={epoch:03d} val_loss={float(val_loss):.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = state.replace(X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stop t={t:02d} best_val={float(best_val):.6f}")
            break

    model_seq[t] = best_state
    return model_seq

# def compute_value_functions(
#     key: jax.Array,
#     config: MainConfig,
#     num_timesteps: int,
#     *,
#     hidden_dims = [256, 256, 256, 256],
#     num_epochs: int = 100,
#     save_debug_data: bool = True,
# ):
#     """
#     Backward recursion to learn a value approximator for each timestep.

#     Returns:
#         model_seq: list/array of model states indexed by timestep.
#     """
#     # input_dim = LOGICAL_PARTICLE_DIM + 1  # +1 for defender action feature
#     # input_dim = LOGICAL_PARTICLE_DIM + config.game.num_defender_actions  # one-hot defender action
#     input_dim = LOGICAL_PARTICLE_DIM + 2 # continous defender action


#     num_particles = config.filter.ADP_samples

#     key, k_init = random.split(key)
#     model_seq = rf.initialize_mlp_sequence(
#         num_stages=num_timesteps,
#         input_dim=input_dim,
#         hidden_dims=HIDDEN_DIMS,
#         rng=k_init,
#     )

#     final_t = num_timesteps - 1


#     # mark terminal stage
#     model_seq[final_t] = model_seq[final_t].replace(is_terminal=True)

#     debug_data_container = {} if save_debug_data else None

#     # # Fit final timestep model first
#     # key, subkey = random.split(key)
#     # p, a, y = adp_compute_data(subkey, config, num_particles, False, model_seq[final_t], timestep=final_t)
#     # model_seq = fit_adp_model_for_timestep(p, a, y, model_seq, config, final_t, num_epochs=num_epochs)

    

#    # (optional) still generate terminal debug data, but DO NOT fit a model
#     if save_debug_data:
#         key, subkey = random.split(key)
#         pT, aT, yT = adp_compute_data(subkey, config, num_particles, False, model_seq[final_t], timestep=final_t)
#         debug_data_container[final_t] = {"particles": pT, "actions": aT, "values": yT}

#     # print_particle_with_labels(get_means(p), config)

#     # Now recurse backwards
#     for t in range(final_t - 1, -1, -1):
#         print(f"Training model for timestep t={t}")
#         key, subkey = random.split(key)
#         p, a, y = adp_compute_data(subkey, config, num_particles, True, model_seq[t + 1], timestep=t)
#         model_seq = fit_adp_model_for_timestep(p, a, y, model_seq, config, t, num_epochs=num_epochs)

#         if save_debug_data:
#             debug_data_container[t] = {
#                 "particles": p,
#                 "actions": a,
#                 "values": y,
#             }

#     if save_debug_data:
#         return model_seq, debug_data_container
#     else:
#         return model_seq

def compute_value_functions(
    key,
    config: MainConfig,
    num_timesteps,
    *,
    hidden_dims,
    activation,
    lr,
    num_epochs=100,
    save_debug_data=False,
):
    """
    Backward recursion to learn a value approximator for each timestep.

    Returns:
        model_seq: list/array of model states indexed by timestep.
    """
    input_dim = LOGICAL_PARTICLE_DIM + 2 # continous defender action

    num_particles = config.filter.ADP_samples

    key, k_init = random.split(key)
    model_seq = rf.initialize_mlp_sequence(
        num_stages=num_timesteps,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        lr=lr,
        rng=k_init,
    )

    final_t = num_timesteps - 1


    # mark terminal stage
    model_seq[final_t] = model_seq[final_t].replace(is_terminal=True)

    debug_data_container = {} if save_debug_data else None

    # # Fit final timestep model first
    # key, subkey = random.split(key)
    # p, a, y = adp_compute_data(subkey, config, num_particles, False, model_seq[final_t], timestep=final_t)
    # model_seq = fit_adp_model_for_timestep(p, a, y, model_seq, config, final_t, num_epochs=num_epochs)

    

   # (optional) still generate terminal debug data, but DO NOT fit a model
    if save_debug_data:
        key, subkey = random.split(key)
        pT, aT, yT = adp_compute_data(subkey, config, num_particles, False, model_seq[final_t], timestep=final_t)
        debug_data_container[final_t] = {"particles": pT, "actions": aT, "values": yT}

    # print_particle_with_labels(get_means(p), config)

    # Now recurse backwards
    # for t in range(final_t - 1, -1, -1):
    for t in trange(
        final_t - 1, -1, -1,
        desc="ADP backward pass",
        leave=False
    ):
        print(f"Training model for timestep t={t}")
        key, subkey = random.split(key)
        p, a, y = adp_compute_data(subkey, config, num_particles, True, model_seq[t + 1], timestep=t)
        model_seq = fit_adp_model_for_timestep(p, a, y, model_seq, config, t, num_epochs=num_epochs)

        if save_debug_data:
            debug_data_container[t] = {
                "particles": p,
                "actions": a,
                "values": y,
            }

        # after fitting timestep t
        # del p, a, y
        # gc.collect()
        # jax.clear_caches()

    if save_debug_data:
        return model_seq, debug_data_container
    else:
        return model_seq

# -----------------------------------------------------------------------------
# Value estimation (max over defender actions)
# -----------------------------------------------------------------------------

def max_value_estimate(
    particle: jnp.ndarray,
    with_model: bool,
    model_state_next,
    config: MainConfig,
) -> jnp.ndarray:
    """
    Return max_d [ cost(s,d) + V_{t+1}(s,d) ] for a single particle.
    """
    particle = particle[:LOGICAL_PARTICLE_DIM]
    defender_actions = jnp.arange(config.game.num_defender_actions)

    def value_with_model(_):
        return jax.vmap(get_value_estimate_with_model, in_axes=(None, 0, None, None))(
            particle, defender_actions, model_state_next, config
        )

    def value_no_model(_):
        return jax.vmap(generate_cost_function_estimate, in_axes=(None, 0, None))(
            particle, defender_actions, config
        )

    values = jax.lax.cond(with_model, value_with_model, value_no_model, operand=None)
    return jnp.max(values)


# def get_value_estimate_with_model(
#     particle: jnp.ndarray,
#     defender_action: int,
#     state_next,
#     config: MainConfig,
# ) -> jnp.ndarray:
#     """
#     Value estimate using:
#         cost(s,d) + V_hat_{t+1}(s,d)
#     where V_hat_{t+1} is the learned MLP for the next timestep.
#     """
#     # # Feature vector: [particle, defender_action]
#     # X = jnp.concatenate([jnp.ravel(particle), jnp.ravel(defender_action)])
#     # Xn = (X - state_next.X_mean.squeeze()) / state_next.X_std.squeeze()

#     num_defender_actions = config.game.num_defender_actions

#     # particle = particle[:LOGICAL_PARTICLE_DIM]
#     # a_oh = _one_hot_def_actions(defender_action, num_defender_actions)  # (K,)
#     # X = jnp.concatenate([jnp.ravel(particle), jnp.ravel(a_oh)])         # (D+K,)

#     # --- Build features (using the continuous action features) ---
#     particle = particle[:LOGICAL_PARTICLE_DIM]

#    # Look up continuous defender action: (2,)
#     a_cont = config.game.defender_action_set[defender_action]

#     # Normalization constants (define once in config if possible)
#     ACCEL_MAX = config.game.max_acceleration    # max |accel|
#     STEER_MAX = config.game.max_steering    # max |steering| (degrees)

#     a_norm = jnp.stack(
#         [
#             a_cont[0] / ACCEL_MAX,
#             a_cont[1] / STEER_MAX,
#         ],
#         axis=0
#     )


#     X = jnp.concatenate([particle, a_norm], axis=0)


#     Xn = (X - state_next.X_mean.squeeze()) / state_next.X_std.squeeze()


#     y_norm = state_next.apply_fn(state_next.params, Xn[None, :]).squeeze()
#     yhat = y_norm * state_next.y_std.squeeze() + state_next.y_mean.squeeze()

#     return generate_cost_function_estimate(particle, defender_action, config) + yhat.squeeze()

def get_value_estimate_with_model(particle, defender_action, state_next, config):
    particle_raw = particle[:LOGICAL_PARTICLE_DIM]

    # transform probability features
    probs = particle_raw[IDX_P1:IDX_P9 + 1]
    log_probs = centered_log_probs(probs)

    particle_feat = particle_raw.at[IDX_P1:IDX_P9 + 1].set(log_probs) 

    a_cont = config.game.defender_action_set[defender_action]
    ACCEL_MAX = config.game.max_acceleration
    STEER_MAX = config.game.max_steering
    a_norm = jnp.stack([a_cont[0] / ACCEL_MAX, a_cont[1] / STEER_MAX], axis=0)

    X = jnp.concatenate([particle_feat, a_norm], axis=0)
    Xn = (X - state_next.X_mean.squeeze()) / state_next.X_std.squeeze()

    # after building X
    assert X.shape[0] == state_next.X_mean.squeeze().shape[0]
    assert X.shape[0] == state_next.X_std.squeeze().shape[0]


    def terminal_case(_):
        return jnp.array(0.0, dtype=X.dtype)

    def nonterminal_case(_):
        y_norm = state_next.apply_fn(state_next.params, Xn[None, :]).squeeze()
        return y_norm * state_next.y_std.squeeze() + state_next.y_mean.squeeze()

    v_next = jax.lax.cond(state_next.is_terminal, terminal_case, nonterminal_case, operand=None)

    return generate_cost_function_estimate(particle_raw, defender_action, config) + v_next


def get_ADP_action_MLP(particle, model_params, defender_actions, config:MainConfig):

    model = rf.MLPRegressor(HIDDEN_DIMS)

    value_estimates = jax.vmap(
        get_value_estimate_with_model, in_axes=(None, 0, None, None)
    )(particle, defender_actions, model_params, config)
    best_action = jnp.argmax(value_estimates)
    return best_action

if __name__ == "__main__":
    key = random.PRNGKey(42)
    key, subkey = random.split(key)
    config = MainConfig.create(subkey)

    model_seq = compute_value_functions(key, config, config.game.num_timesteps)
