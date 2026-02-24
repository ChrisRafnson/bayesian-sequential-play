import jax.numpy as jnp

"""
constants.py

Stores the commonly used constants such as particle indices to be used across files in the rest of the repository.
"""

# Predefine a single "particle"
MAX_PARTICLE_DIM = 32
LOGICAL_PARTICLE_DIM = 17
NUM_PARTICLES = 100_000
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