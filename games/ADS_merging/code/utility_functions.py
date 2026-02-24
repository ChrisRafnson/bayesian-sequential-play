"""
utility_functions.py

Contains the implementations of the utility functions used by the attacker and defender.
"""

import jax.numpy as jnp
from config import MainConfig
import jax
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def shaped_peak(value, peak=0.0, width=1.0, reward_scale=1.0, penalty_scale=1.0):
    """
    Custom utility shaping function:
    - Max value at 'peak'
    - Smoothly decays to negative penalty
    - Flatlines beyond width*3 to avoid extreme penalties
    """
    deviation = value - peak
    abs_dev = jnp.abs(deviation / width)

    # High utility near peak, drops off quickly, then saturates
    shaped_value = reward_scale * jnp.exp(-abs_dev**2) - penalty_scale * jnp.tanh(abs_dev)

    return shaped_value


def utility(
    true_vertical,
    observed_horizontal,
    observed_vertical,
    observed_heading,
    observed_speed,
    observed_opponent_horizontal,
    observed_opponent_vertical,
    observed_opponent_heading,
    observed_opponent_speed,
    action,
    beta_1,  # off-road penalty
    beta_2,  # car proximity
    beta_3,  # horiz proximity range
    beta_4,  # vert proximity range
    beta_5,  # speed penalty
    beta_6,  # idle utility
    config: MainConfig,
    is_defender: bool,
    beta_7=2.0
):  

    soft_merge = config.game.W1
    hard_merge = config.game.W2

    # First calculate their merge progress 0 if they have not passed the soft merge point
    # and 1 if they are past the hard merge point. Otherwise it is a representation of the percentage
    # of distance they have covered between the soft and hard merge thresholds. 
    
    merge_progress = (observed_vertical - soft_merge)/(hard_merge - soft_merge)
    merge_progress = jnp.clip(merge_progress, min=0.0, max=1.0)


    #Now we have the bounds of the road, the right bound shrinks starting at the soft merge
    #point, eventually becoming a one lane road
    left_bound = config.game.l1
    middle = config.game.l2
    right_bound = (1-merge_progress)*config.game.l3 + merge_progress * middle

    # =================== Proximity Utility ============================

    dx = observed_horizontal - observed_opponent_horizontal # horizontal difference between centroids
    dy = observed_vertical - observed_opponent_vertical # vertical difference between centroids

    horizontal_buffer = config.game.car_width + beta_3 #The allowable horizontal distance between centroids, accounts for 2 half-widths and the player's safety buffer
    vertical_buffer = config.game.car_length + beta_4 #The allowable vertical distance between centroids, accounts for 2 half-lengths and the player's buffer

    horizontal_violation = jnp.maximum(0, (1-jnp.abs(dx)/horizontal_buffer))
    vertical_violation = jnp.maximum(0, (1-jnp.abs(dy)/vertical_buffer))

    combined_violation = horizontal_violation * vertical_violation

    proximity_penalty = -beta_5 * combined_violation

    # =================== Boundary Utility ============================

    #This part penalizes the agents for approach the road boundaries and going off the road as well

    road_margin = config.game.road_margin  # e.g. 0.25–0.5 meters

    # Signed clearance to each wall (0 = safe)
    d_left  = jnp.minimum(0, observed_horizontal - (left_bound + road_margin))
    d_right = jnp.minimum(0, (right_bound - road_margin) - observed_horizontal)

    # Logistic wall penalties in (0, 1)
    left_wall_pen  = jax.nn.softplus(-d_left * beta_1)
    right_wall_pen = jax.nn.softplus(-d_right * beta_1)

    # Total bounded boundary utility
    boundary_utility = -jnp.abs(left_wall_pen + right_wall_pen)

    # ================== Comfort Utility ===================

    # how far off-road are we?
    offroad = jnp.maximum(
        left_bound + road_margin - observed_horizontal,
        observed_horizontal - (right_bound - road_margin),
    )
    offroad = jnp.maximum(offroad, 0.0)

    # steering penalty fades out when off-road
    steering_penalty = -beta_2 * (1.0 - jnp.tanh(offroad)) * (jnp.abs(observed_heading) / config.game.max_heading)

    # ================== Total Utility ==============================

    #Total Utility is just the sum of the utility components

    total_utility = boundary_utility + steering_penalty + proximity_penalty

    return total_utility


def utility_debugger(
    true_vertical,
    observed_horizontal,
    observed_vertical,
    observed_heading,
    observed_speed,
    observed_opponent_horizontal,
    observed_opponent_vertical,
    observed_opponent_heading,
    observed_opponent_speed,
    beta_1,  # off-road penalty
    beta_2,  # car proximity
    beta_3,  # horiz proximity range
    beta_4,  # vert proximity range
    beta_5,  # speed penalty
    beta_6,  # navigation utility
    config: MainConfig,
    ):
    """
    Defender utility function with history-dependent penalties (summation form, JAX compatible).

    Args:
        attacker_action (int): Current attacker action (index form).
        defender_action (int): Current site left undefended.
        config (MainConfig): Game configuration.

    Returns:
        utility (float): Defender utility for this timestep.
    """
    #Get the lane and merge boundaries
    l1 = config.game.l1
    l2 = config.game.l2
    l3 = config.game.l3

    road_width = l3-l1

    W1 = config.game.W1
    W2 = config.game.W2

    S1 = config.game.S1 # Minimum legal speed
    S2 = config.game.S2 # Maximum legal speed

    # Road boundaries depend on y:
    #   before hard merge (y < W2): road is [l1, l3]
    #   after hard merge (y >= W2): road is [l1, l2]  (single lane)
    right_bound = jnp.where(observed_vertical < W2, l3, l2)

    # ----------- Off-Roading Penalty ---------------------------------------------------------------------------------------

    off_dist = jnp.maximum(l1 - observed_horizontal, 0.0) + jnp.maximum(observed_horizontal - l3, 0.0)
    off_norm = off_dist / (road_width + 1e-6)
    off_roading_penalty = -beta_1 * jnp.tanh(off_norm)**2

    dist_left  = jnp.maximum(l1 - observed_horizontal, 0.0)
    dist_right = jnp.maximum(observed_horizontal - l3, 0.0)
    dist_off = dist_left + dist_right  # meters off-road

    off_roading_penalty = -beta_1 * (dist_off ** 2)


    # ----------- Heading Penalty ---------------------------------------------------------------------------------------

    # Assume heading = observed_speed[1]  # or however heading is represented
    # heading_penalty = -beta_1 * (observed_heading ** 2) /45

    #----------- Car Proximity Penalty --------------------------------------------------------------------------------------

    #Calculate distances between car centroids
    car_horiz_diff = jnp.abs(observed_horizontal-observed_opponent_horizontal) - 2 #Subtract 2 half-widths of the cars to get horizontal distance between chassis
    car_vert_diff = jnp.abs(observed_vertical-observed_opponent_vertical) - 4.5 #Subtract 2 half-lenghts of the cars to get vertical distance between chassis
    car_horiz_diff = jnp.maximum(car_horiz_diff, 0.0) #Clip values for stability
    car_vert_diff  = jnp.maximum(car_vert_diff, 0.0)  #Clip values for stability

    # normalized closeness in [0, 1] (1 = very close, 0 = far)
    cx = jnp.maximum(beta_3 - car_horiz_diff, 0.0) / beta_3
    cy = jnp.maximum(beta_4 - car_vert_diff, 0.0) / beta_4

    danger_level = cx * cy              # only >0 if BOTH are within thresholds
    car_proximity_penalty = beta_2 * -danger_level**2         # or -too_close, or -k*too_close

    #----------- Speed Violation Penalty --------------------------------------------------------------------------------------

    low = jnp.maximum((S1 - observed_speed) / (S1), 0.0)
    high = jnp.maximum((observed_speed - S2) / (S2), 0.0)
    speed_violation_penalty = -beta_5 * (low + high)


    #----------- Navigation Utility --------------------------------------------------------------------------------------


    #Next we calculate the navigation objective, which is a piecewise function, we will use
    #boolean operators to apply the various conditions

    in_left_lane = (observed_horizontal > l1) & (observed_horizontal < l2) & (observed_vertical < W1)
    in_right_lane = (observed_horizontal > l2) & (observed_horizontal < l3) & (observed_vertical < W1)
    passed_soft_merge = (observed_vertical >= W1)

    left_lane_util = -(observed_horizontal - l1) * (observed_horizontal - l2) * in_left_lane.astype(float)
    right_lane_util = -(observed_horizontal - l2) * (observed_horizontal - l3) * in_right_lane.astype(float)

    z = (true_vertical - W1) / (W2 - W1)
    z = jnp.clip(z, 0.0, 1.0)          # cap once you reach W2
    urgency = 1.0 + z**2               # in [1, 2]


    urgent_merge_util = -(observed_horizontal - l1) * (observed_horizontal - l2) * urgency * passed_soft_merge.astype(float)

    navigation_obj = left_lane_util + right_lane_util + urgent_merge_util
    navigation_utility =  beta_6 * navigation_obj

    total_utility = navigation_utility + speed_violation_penalty + car_proximity_penalty + off_roading_penalty

    # total_utility = jnp.clip(total_utility, -200, 100)


    return total_utility, off_roading_penalty, speed_violation_penalty,  car_proximity_penalty, navigation_utility

def get_action_components(action, is_defender, config):
    def defender_branch(_):
        return config.game.defender_action_set[action]

    def attacker_branch(_):
        return config.game.attacker_action_set[action]

    return jax.lax.cond(
        is_defender,
        defender_branch,
        attacker_branch,
        operand=None,
    )

if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # Minimal config stub (replace with your real MainConfig if available)
    # --------------------------------------------------------------------

    seed = 1

    np.random.seed(seed)  # Set seed
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    config = MainConfig.create(subkey)

    # Fixed parameters
    params = dict(
        true_vertical=0.0,
        observed_heading=0.0,
        observed_speed=20.0,
        observed_opponent_horizontal=0.0,
        observed_opponent_vertical=70.0,
        observed_opponent_heading=0.0,
        observed_opponent_speed=20.0,
        beta_1=1.0,
        beta_2=0.0,
        beta_3=2.0,
        beta_4=2.0,
        beta_5=0.0,
        beta_6=1.0,
        config=config,
    )

    util = utility(
        observed_horizontal=6,
        observed_vertical=81.20,
        **params
    )

    print(util)