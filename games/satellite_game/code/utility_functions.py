"""
utility_functions.py

Contains the implementations of the utility functions used by the attacker and defender.
"""

import jax.numpy as jnp

def get_utility(betas, observed_state, action):
    """
    The utility function to return the payoffs for a player based on their observed distance.
    This version is paramterized with an action to allow for utilities based upon action and observed state.
    Because we want to penalize actions 1 and 3

    Args:
        betas (array): The parameters for the player's unique payoff structure
        obsevered_distance (float): The player's observed distance from the other player.
        action (int): The player's action taken

    Returns:
        utility (float): The utility for the observed distance
    """

    center = (betas[0] + betas[1]) / 2
    span = betas[1] - betas[0]
    scaled = (observed_state - center) / span
    reward = betas[2] * (1 - (2 * scaled) ** 2)
    penalty = jnp.where(action == 1, 0.0, betas[3]) / (
        scaled**2 + 0.0001
    )  # no penalty if action == 1, added 0.0001 to ensure no division by zero
    utility = reward - penalty
    return utility
