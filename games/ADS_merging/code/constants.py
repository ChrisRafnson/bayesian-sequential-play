"""
constants.py

Stores the commonly used constants such as particle indices to be used across files in the rest of the repository.
"""

import jax.numpy as jnp

# Predefine a single "particle"
MAX_PARTICLE_DIM = 256
MAX_NUM_ATTACKER_ACTIONS = 9 #2 selections from 3 vectors of size 3, total of 9 actions
LOGICAL_PARTICLE_DIM = 38
NUM_PARTICLES = 10_000

#Experience Parameter
IDX_N = 0 

#EWA Parameters
IDX_DELTA = 1
IDX_PHI = 2
IDX_RHO = 3
IDX_LAMBDA = 4

#Beta payoff parameters for the attacker
IDX_BETA_A1 = 5
IDX_BETA_A2 = 6
IDX_BETA_A3 = 7
IDX_BETA_A4 = 8
IDX_BETA_A5 = 9
IDX_BETA_A6 = 10

#Attacker Noise Parameter
IDX_SIGMA = 11

# Latent State parameters X_t

#Defender Variables
IDX_XD1 = 12 # horizontal position of the defender centroid
IDX_XD2 = 13 # vertical position of the defender centroid
IDX_XD3 = 14 # heading angle of the defender
IDX_XD4 = 15 # speed of the defender

#Attacker Variables
IDX_XA1 = 16 # horizontal position of the attacker centroid
IDX_XA2 = 17 # vertical position of the attacker centroid
IDX_XA3 = 18 # heading angle of the attacker
IDX_XA4 = 19 # speed of the attacker

"""
Attractions and Probabilities of play.

Allowing the attacker to attack the same site twice results in 100 possible actions.

Allowing the attacker to only attack each site once results in 45 possible actions.

We assume that action 1 is associated with sites [1, 1], action 2 with sites [1, 2]...
"""

IDX_A1 = 20 #Index of the first attraction
IDX_A9 = IDX_A1 + 8 #Index of the last attraction
IDX_P1 = IDX_A9 + 1 #Index of the probability of the first action
IDX_P9 = IDX_P1 + 8 #Index of the probability of the last action

PARAMETER_INDICES = jnp.array(
    [IDX_DELTA, IDX_PHI, IDX_RHO, IDX_LAMBDA, IDX_N]
)  # Tells JAX where to update the parameters, delta, phi, rho, lambda, and the experience
BETA_INDICES = jnp.array([IDX_BETA_A1, IDX_BETA_A2, IDX_BETA_A3, IDX_BETA_A4, IDX_BETA_A5, IDX_BETA_A6])  # Tells JAX where to update the betas

#MLP Settings

HIDDEN_DIMS = [256, 256, 256, 256]
HUBER_DELTA = 1.5 #For Huber loss