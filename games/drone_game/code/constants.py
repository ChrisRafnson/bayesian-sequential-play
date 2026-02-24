"""
constants.py

Stores the commonly used constants such as particle indices to be used across files in the rest of the repository.
"""

import jax.numpy as jnp



# Predefine a single "particle"
MAX_PARTICLE_DIM = 256
MAX_NUM_ATTACKER_ACTIONS = 100 #2 selections from 10 sites = 100 possible actions. This allows us to define blocks for the attractions a probs
LOGICAL_PARTICLE_DIM = 209
NUM_PARTICLES = 10000


#EWA Parameters
IDX_DELTA = 0
IDX_PHI = 1
IDX_RHO = 2
IDX_LAMBDA = 3

#Beta payoff parameter for the attacker
IDX_BETA = 4

#Experience Parameter
IDX_N = 5

# Latent State parameters X_t, either NULL/-1 or A_t and D_t
IDX_A = 6
IDX_D = 7
IDX_T = 8 #The current timestep


"""
Attractions and Probabilities of play.

Allowing the attacker to attack the same site twice results in 100 possible actions.

Allowing the attacker to only attack each site once results in 45 possible actions.

We assume that action 1 is associated with sites [1, 1], action 2 with sites [1, 2]...
"""

IDX_A1 = 9 #Index of the first attraction
IDX_A100 = IDX_A1 + 99 #Index of the last attraction
IDX_P1 = IDX_A100 + 1 #Index of the probability of the first action
IDX_P100 = IDX_P1 + 99 #Index of the probability of the last action




PARAMETER_INDICES = jnp.array(
    [IDX_DELTA, IDX_PHI, IDX_RHO, IDX_LAMBDA, IDX_N]
)  # Tells JAX where to update the parameters, delta, phi, rho, lambda, and the experience
BETA_INDICES = jnp.array([IDX_BETA])  # Tells JAX where to update the betas

#MLP Settings

HIDDEN_DIMS = [64, 32]

#Graph Definitions, padding is so they behave in JAX

SITE_ROUTES = jnp.array([
    jnp.array([0, 1, 6, 7, -1, -1, -1, -1]), #Neighbors of Node 1
    jnp.array([1, 0, -1, -1, -1, -1, -1, -1]), #Neighbors of Node 2
    jnp.array([2, 4, 5, -1, -1, -1, -1, -1]), #Neighbors of Node 3
    jnp.array([3, 6, -1, -1, -1, -1, -1, -1]), #Neighbors of Node 4
    jnp.array([4, 2, 6, -1, -1, -1, -1, -1]), #Neighbors of Node 5
    jnp.array([5, 2, 6, -1, -1, -1, -1, -1]), #Neighbors of Node 6
    jnp.array([6, 0, 3, 4, 5, -1, -1, -1]), #Neighbors of Node 7
    jnp.array([7, 0, -1, -1, -1, -1, -1, -1]), #Neighbors of Node 8
])

ALL_SITES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7])