"""
utility_functions.py

Contains the implementations of the utility functions used by the attacker and defender.
"""

import jax.numpy as jnp
import jax
from config import MainConfig
from utils import ij_to_index, index_to_ij
from constants import SITE_ROUTES, ALL_SITES

# JAX-safe conditional mask
def mask_single(_):
    # Only second element valid
    return jnp.array([False, True])

def mask_double(_):
    # Both valid
    return jnp.array([True, True])

def site_weight(site_index, timestep, c=10, period=9):
    return c * (jnp.cos(timestep + site_index) ** 2) #V1 weighting function

    # return c * jnp.sign(jnp.cos(timestep + site_index)) + c #V2 weighting function

    # return c * jnp.sign(jnp.cos(timestep + site_index)) + c #V3 weighting function

    # #V6 weighting function
    # phase = timestep % period         # value goes 0 → 8 then loops 
    # step = period // 3         # step = 3

    # return jnp.where(phase < step, 0.5,
    #        jnp.where(phase < 2 * step, 1, 2))

def defender_utility(
        attacker_action,
        defender_action,
        config:MainConfig,
        timestep,
        prev_attacker_action = -1,
        prev_defender_action = -1
    ):
    """
    Defender utility function with history-dependent penalties (summation form, JAX compatible).

    Args:
        attacker_action (int): Current attacker action (index form).
        defender_action (int): Current site left undefended.
        prev_attacker_action (int): Previous attacker action (index form, -1 if none).
        prev_defender_action (int): Previous undefended site (-1 if none).
        config (MainConfig): Game configuration.
        c (float): Scaling parameter for repeat punishments.

    Returns:
        utility (float): Defender utility for this timestep.
    """

    a = config.game.utility_a
    b = config.game.utility_b
    c = config.game.utility_c

    # --- Travel Cost ---

    """ 
    This is the penalty associated with taking selecting a site that is not a 
    neighbor of the previous undefended site. The not operator ~ negates the outcome
    of jnp.isin(), so we get TRUE if the chosen site is not in the neighbors of the previous
    chosen site
    """

    # neighbors = jax.lax.cond(
    #     prev_defender_action == -1,
    #     lambda _ :ALL_SITES,
    #     lambda _ :SITE_ROUTES[jnp.int16(prev_defender_action)],
    #     operand=None
    # )

    neighbors = jnp.where(
        prev_defender_action == -1,
        ALL_SITES,
        SITE_ROUTES[jnp.array(prev_defender_action, int)])

    # neighbors = jax.lax.cond(
    # prev_defender_action == -1,
    # lambda _: ALL_SITES,   # shape must match!
    # lambda _: jax.lax.dynamic_index_in_dim(
    #     SITE_ROUTES, prev_defender_action, axis=0
    # ),
    # None
    # )

    TC  = -a * (~jnp.isin(defender_action, neighbors))

    # --- Priority Cost ---

    """
    This gives us a penalty proportional to the site's priority weighting at the time.
    This should incentivize us to not leave high priority targets undefended.
    """
    PC = -b * site_weight(defender_action, timestep)

    # --- Attack Cost ---

    """
    This just applies a penalty if the undefended site is the one attacked by the opponent
    """
    AC = -c * (defender_action == attacker_action)

    return TC + PC + AC


def attacker_utility(attacker_action, defender_action, beta, config: MainConfig, timestep):
    """
    The utility function for the attacker. The attacker receives beta utility if their chosen sites are
    undefended by the defender

    Args:
        attacker_actions (int): Attacker action in index form
        defender_action (int): The site chosen to be left undefended.
        beta (float): The attacker's scaling parameter
        

    Returns:
        utility (float): The utility for the chosen actions
    """

    n = config.game.num_sites
    nas = config.game.num_attacker_selections

    # [i, j] always length 2
    attacker_action_array = index_to_ij(attacker_action, n=n)

    valid_mask = jax.lax.cond(
        nas == 1,
        mask_single,
        mask_double,
        operand=None
    )

    # Apply mask (ignore unused slot if single-site)
    hits = (defender_action == attacker_action_array) & valid_mask
    
    # By summing across the mask [True, False] --> 1 and [True, True] --> 2 it gives the attacker 2 * beta
    # utility if the undefended site is attacked twice. This can be adjusted if we want.
    utility = beta * jnp.sum(hits) 

    return utility

if __name__ == "__main__":

    defender_action = 5
    attacker_action = 1

    prev_def_act = -1
    prev_atk_act = -1

    config = MainConfig.create(jax.random.PRNGKey(42), game_kwargs={"num_attacker_selections": 1})

    util = defender_utility(attacker_action, defender_action, config, timestep=0, prev_defender_action=prev_def_act, prev_attacker_action=prev_atk_act)
    print(f"Defender Utility: {util}")

    util = attacker_utility(attacker_action, defender_action, 3.81, config, timestep=0)
    print(f"Attacker Utility: {util}")

    for i in range(30):
        print(site_weight(0, i))


