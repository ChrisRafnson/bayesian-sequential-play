import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from utility_functions import utility

import jax.numpy as jnp
import jax.random as random
from flax import (  # Using the flax.struct.dataclass for compatibility with jax.jit
    struct,
)
from constants import NUM_PARTICLES
import jax.numpy as jnp
from pprint import pformat
from dataclasses import field


# ---------------------------------------------------------------------
# Minimal config stub (replace with your real MainConfig if available)
# ---------------------------------------------------------------------
@dataclass
class GameConfig:

    # error_X_sigma: float = 0.0 #Noise for all latent state vars
    # error_Y_sigma: float = 0.000001 #Noise for all defender observations

    error_X_sigma: jnp.ndarray = field(
        default_factory=lambda: jnp.array([
            0.0,   # XD1 (m)
            0.0,   # XD2 (m)
            0.0,  # XD3 (rad) ~ 1.7°
            0.0,   # XD4 (m/s)
            0.0,   # XA1
            0.0,   # XA2
            0.0,  # XA3
            0.0,   # XA4
        ], dtype=jnp.float32),
    )

    error_Y_sigma: jnp.ndarray = field(
        default_factory=lambda: jnp.array([
            0.25,   # XD1
            0.5,   # XD2
            0.5,  # XD3
            0.5,   # XD4
            0.25,   # XA1
            0.5,   # XA2
            0.5,  # XA3
            0.5,   # XA4
        ], dtype=jnp.float32),
    )

    XD1_init: float = 6.0 #Starting value for XD1 (meters)
    XD2_init: float = 0.0 #Starting value for XD2 (meters)
    XD3_init: float = 0.0 #Starting value for XD3 (degrees)
    XD4_init: float = 22.0 #Starting value for XD4 (m/s)

    XA1_init: float = 2.0 #Starting value for XA1 (meters)
    XA2_init: float = 0.0 #Starting value for XA2 (meters)
    XA3_init: float = 0.0 #Starting value for XA3 (degrees)
    XA4_init: float = 27.0 #Starting value for XA4 (m/s)

    #Dimensions are in METERS!

    L: float = 2.0 #The interaxle distance
    car_length:float = 4.0 #The length of the cars
    car_width: float = 2.0 #The width of the cars


    l1: float = 0.0 #The outer boundary of the left lane
    l2: float = 4.0 #The centerline
    l3: float = 8.0 #The outer boundary of the right lane

    W1: float = 20.0 #The soft merge point
    W2: float = 80.0 #The hard merge point, only one lane hereafter

    S1: float = 20.0 #The minimum legal speed for the road
    S2: float = 30.0 #The speed limit for the road

    delta_t: float = 0.1 #How long the time intervals are (seconds)
    num_timesteps: int = 40 # How long the game will be played for
    num_attacker_actions: int = 9
    num_defender_actions: int = 9

    defender_beta_1: float = 1.0
    defender_beta_2: float = 1.0
    defender_beta_3: float = 1.0
    defender_beta_4: float = 2.0
    defender_beta_5: float = 1.0
    defender_beta_6: float = 0.0

    max_acceleration: float = 2.0
    max_steering: float = 3.0
    max_heading: float = 3.0 #Ideally what we want our maximum heading in either direction to be
    road_margin: float = 0.75 #How close the agents can get to the road boundary before incurring a penalty
    wall_width: float = 1.0 #The steepness of the boundary penalty
    slot_gap: float = 4.0

    

    #Defender Actions - A vector storing tuple pairs of form (D1, D2) where D1 is acceleration (m/s) and D2 is steering angle (degrees)
    defender_action_set: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [
                (-2.0, -1.0),
                (-2.0,  0.0),
                (-2.0,  1.0),
                ( 0.0, -1.0),
                ( 0.0,  0.0),
                ( 0.0,  1.0),
                ( 2.0, -1.0),
                ( 2.0,  0.0),
                ( 2.0,  1.0),
            ],
            dtype=jnp.float32,
        )
    )


    #Attacker Actions - A vector storing tuple pairs of form (D1, D2) where D1 is acceleration (m/s) and D2 is steering angle (degrees)
    attacker_action_set: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [
                (-2.5, -1.0),
                (-2.5,   0.0),
                (-2.5,  1.0),
                ( 0.0, -1.0),
                ( 0.0,   0.0),
                ( 0.0,  1.0),
                (2.0, -1.0),
                (2.0,   0.0),
                (2.0,  1.0),
            ],
            dtype=jnp.float32,
        )
    )

@dataclass
class MainConfig:
    game: GameConfig

config = MainConfig(game=GameConfig())

# ---------------------------------------------------------------------
# Utility function (PASTE YOUR EXACT FUNCTION HERE)
# ---------------------------------------------------------------------
# def utility_test(
#     true_vertical,
#     observed_horizontal,
#     observed_vertical,
#     observed_heading,
#     observed_speed,
#     observed_opponent_horizontal,
#     observed_opponent_vertical,
#     observed_opponent_heading,
#     observed_opponent_speed,
#     beta_1,  # off-road penalty
#     beta_2,  # car proximity
#     beta_3,  # horiz proximity range
#     beta_4,  # vert proximity range
#     beta_5,  # speed penalty
#     beta_6,  # navigation utility
#     config: MainConfig,
# ):
#     """
#     Defender utility function with history-dependent penalties (summation form, JAX compatible).

#     Args:
#         attacker_action (int): Current attacker action (index form).
#         defender_action (int): Current site left undefended.
#         config (MainConfig): Game configuration.

#     Returns:
#         utility (float): Defender utility for this timestep.
#     """
#     #Get the lane and merge boundaries
#     l1 = config.game.l1
#     l2 = config.game.l2
#     l3 = config.game.l3

#     road_width = l3-l1

#     W1 = config.game.W1
#     W2 = config.game.W2

#     S1 = config.game.S1 # Minimum legal speed
#     S2 = config.game.S2 # Maximum legal speed

#     # Road boundaries depend on y:
#     #   before hard merge (y < W2): road is [l1, l3]
#     #   after hard merge (y >= W2): road is [l1, l2]  (single lane)
#     right_bound = jnp.where(observed_vertical < W2, l3, l2)

#     # ----------- Off-Roading Penalty ---------------------------------------------------------------------------------------
#     # Road center depends on merge phase
#     road_center = jnp.where(
#         observed_vertical < W2,
#         0.5 * (l1 + l3),   # two-lane center
#         0.5 * (l1 + l2),   # single-lane center
#     )

#     road_half_width = jnp.where(
#         observed_vertical < W2,
#         0.5 * (l3 - l1),
#         0.5 * (l2 - l1),
#     )

#     # Signed normalized deviation
#     x_dev = (observed_horizontal - road_center) / road_half_width

#     off_roading_penalty = -beta_1 * (
#         jnp.abs(x_dev) + 0.5 * x_dev**2
#     )



#     # ----------- Heading Penalty ---------------------------------------------------------------------------------------

#     # Assume heading = observed_speed[1]  # or however heading is represented
#     # heading_penalty = -beta_1 * (observed_heading ** 2) /45

#     #----------- Car Proximity Penalty --------------------------------------------------------------------------------------

#     # #Calculate distances between car centroids
#     # car_horiz_diff = jnp.abs(observed_horizontal-observed_opponent_horizontal) - 2 #Subtract 2 half-widths of the cars to get horizontal distance between chassis
#     # car_vert_diff = jnp.abs(observed_vertical-observed_opponent_vertical) - 4.5 #Subtract 2 half-lenghts of the cars to get vertical distance between chassis
#     # car_horiz_diff = jnp.maximum(car_horiz_diff, 0.0) #Clip values for stability
#     # car_vert_diff  = jnp.maximum(car_vert_diff, 0.0)  #Clip values for stability

#     # # normalized closeness in [0, 1] (1 = very close, 0 = far)
#     # cx = jnp.maximum(beta_3 - car_horiz_diff, 0.0) / beta_3
#     # cy = jnp.maximum(beta_4 - car_vert_diff, 0.0) / beta_4

#     # danger_level = cx * cy              # only >0 if BOTH are within thresholds
#     # car_proximity_penalty = beta_2 * -danger_level**2         # or -too_close, or -k*too_close

#     #----------- Speed Violation Penalty --------------------------------------------------------------------------------------

#     # low = jnp.maximum((S1 - observed_speed) / (S1), 0.0)
#     # high = jnp.maximum((observed_speed - S2) / (S2), 0.0)
#     # speed_violation_penalty = -beta_5 * (low + high)


#     #----------- Navigation Utility --------------------------------------------------------------------------------------


#     total_utility =  off_roading_penalty

#     # total_utility = jnp.clip(total_utility, -200, 100)

#     return total_utility

#     # return off_roading_penalty + speed_violation_penalty + car_proximity_penalty + navigation_utility


# ---------------------------------------------------------------------
# Plot utility over (x, y)
# ---------------------------------------------------------------------
def plot_utility_surface():

    # Grid
    x = np.linspace(-10.0, 10.0, 300)      # observed_horizontal
    y = np.linspace(0.0, 100.0, 300)     # observed_vertical
    X, Y = np.meshgrid(x, y)

    # Fixed parameters
    params = dict(
        true_vertical=60.0,
        observed_heading=0.0,
        observed_speed=20.0,
        observed_opponent_horizontal=0.0,
        observed_opponent_vertical=70.0,
        observed_opponent_heading=0.0,
        observed_opponent_speed=20.0,
        action=4,
        beta_1=10.0,
        beta_2=5.0,
        beta_3=2.0,
        beta_4=2.0,
        beta_5=5.0,
        beta_6=5.0,
        config=config,
        is_defender=True
    )

    util_fn = jax.vmap(
    lambda x, y: utility(
        observed_horizontal=x,
        observed_vertical=y,
        **params
    ),
    in_axes=(0, 0),
)

    U = np.array(util_fn(X.flatten(), Y.flatten())).reshape(X.shape)


    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.contourf(X, Y, U, levels=50)
    # plt.colorbar(label="Utility")
    # plt.xlabel("Observed Horizontal Position")
    # plt.ylabel("Observed Vertical Position")
    # plt.title("Defender Utility Landscape")
    # plt.savefig("utilitysurface.png")
    # # plt.show()

    plt.figure(figsize=(10, 6))

    # Filled contours (smooth background)
    cf = plt.contourf(
        X, Y, U,
        levels=100,          # smoother fill
    )

    # Contour lines (structure)
    cs = plt.contour(
        X, Y, U,
        levels=100,           # fewer, clearer lines
        linewidths=0.8,
        colors="black",
        alpha=0.6,
    )

    # Optional: label contour lines
    plt.clabel(
        cs,
        inline=True,
        fontsize=8,
        fmt="%.1f",
    )

    plt.colorbar(cf, label="Utility")

    plt.xlabel("Observed Horizontal Position")
    plt.ylabel("Observed Vertical Position")
    plt.title("Defender Utility Landscape")

    plt.tight_layout()
    plt.savefig("utilitysurface.png", dpi=300)
    # plt.show()


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
plot_utility_surface()
