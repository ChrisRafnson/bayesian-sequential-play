import jax.numpy as jnp
import jax.random as random
from flax import (  # Using the flax.struct.dataclass for compatibility with jax.jit
    struct,
)
from constants import NUM_PARTICLES
import jax.numpy as jnp
from pprint import pformat
from dataclasses import field

@struct.dataclass
class GameConfig:
    error_Z_sigma: float #Unique for each instance

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
            0.2,   # XD1
            0.5,   # XD2
            0.5,  # XD3
            0.5,   # XD4
            0.2,   # XA1
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

    defender_beta_1: float = 3.0
    defender_beta_2: float = 5.0
    defender_beta_3: float = 1.0
    defender_beta_4: float = 2.0
    defender_beta_5: float = 5.0
    defender_beta_6: float = 0.0

    max_acceleration: float = 2.0
    max_steering: float = 1.0
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


    @staticmethod
    def create(key, **kwargs):

        # Parameters for drawing the error_Z_variance
        IG_alpha: int = 2
        IG_beta: int = 1
        error_Z_sigma_scalar = IG_beta / random.gamma(key, IG_alpha)
        

        return GameConfig(error_Z_sigma=0, **kwargs)

@struct.dataclass
class PriorConfig:

    # Note: The parameters refer to the mean and scale parameters for a beta distribution.
    # The bounds refer to the limits of the beta distribution
    AEP_scale: int = 10

    delta_parameters: tuple[float, float] = (1, 1)
    delta_bounds: tuple[int, int] = (0, 1)

    phi_parameters: tuple[float, float] = (1, 1)
    phi_bounds: tuple[int, int] = (0, 1)

    rho_parameters: tuple[float, float] = (1, 1)
    rho_bounds: tuple[int, int] = (0, 1)

    lambda_parameters: tuple[float, float] = (1, 1)
    # lambda_bounds: tuple[int, int] = (1, 3)
    lambda_bounds: tuple[int, int] = (0.5, 2)

    experience_parameters: tuple[float, float] = (1, 1)
    experience_bounds: tuple[int, int] = (1, 30)

    # initial_attraction_parameters: tuple[float, float] = (1, 1)
    # initial_attraction_bounds: tuple[int, int] = (0, 5)

    initial_attraction_alpha: jnp.ndarray = field(default_factory=lambda: jnp.array([1,1,1, 1,1,1, 1,1,1,], dtype=jnp.float32))
    initial_attraction_beta:  jnp.ndarray = field(default_factory=lambda: jnp.array([1,1,1, 1,1,1, 1,1,1,], dtype=jnp.float32))
    initial_attraction_lower: jnp.ndarray = field(default_factory=lambda: jnp.array([-5,-5,-5, -5,-5,-5, -5,-5,-5], dtype=jnp.float32))
    initial_attraction_upper: jnp.ndarray = field(default_factory=lambda: jnp.array([5,5,5, 5,5,5, 5,5,5], dtype=jnp.float32))

    sigma_parameters: tuple[float, float] = (1, 1)
    sigma_bounds: tuple[int, int] = (0, 5)

    beta_A1_parameters: tuple[float, float] = (1, 1)
    beta_A1_bounds: tuple[int, int] = (0, 5)

    beta_A2_parameters: tuple[float, float] = (1, 1)
    beta_A2_bounds: tuple[int, int] = (0, 5)

    beta_A3_parameters: tuple[float, float] = (1, 1)
    beta_A3_bounds: tuple[int, int] = (0, 5)

    beta_A4_parameters: tuple[float, float] = (1, 1)
    beta_A4_bounds: tuple[int, int] = (0, 5)

    beta_A5_parameters: tuple[float, float] = (1, 1)
    beta_A5_bounds: tuple[int, int] = (0, 5)

    beta_A6_parameters: tuple[float, float] = (1, 1)
    beta_A6_bounds: tuple[int, int] = (0, 5)

    XD1_parameters: tuple[float, float] = (4, 4)
    # XD1_bounds: tuple[int, int] = (-8, 16) #Horizontal Pos
    XD1_bounds: tuple[int, int] = (4, 8) #Horizontal Pos

    XD2_parameters: tuple[float, float] = (1, 1)
    # XD2_bounds: tuple[int, int] = (0, 200) #Vertical Pos
    XD2_bounds: tuple[int, int] = (0, 5) #Vertical Pos

    XD3_parameters: tuple[float, float] = (2, 2)
    XD3_bounds: tuple[int, int] = (-2, 2) #Heading

    XD4_parameters: tuple[float, float] = (1, 1)
    XD4_bounds: tuple[int, int] = (20, 30) #Speed

    XA1_parameters: tuple[float, float] = (4, 4)
    # XA1_bounds: tuple[int, int] = (-8, 16) #Horizontal Pos
    XA1_bounds: tuple[int, int] = (0, 4) #Horizontal Pos

    XA2_parameters: tuple[float, float] = (1, 1)
    # XA2_bounds: tuple[int, int] = (0, 200) #Vertical Pos
    XA2_bounds: tuple[int, int] = (0, 5) #Vertical Pos


    XA3_parameters: tuple[float, float] = (2, 2)
    XA3_bounds: tuple[int, int] = (-2, 2) #Heading

    XA4_parameters: tuple[float, float] = (1, 1)
    XA4_bounds: tuple[int, int] = (20, 30) #Speed

    epsilon_parameters: tuple[float, float] = (1, 1)
    epsilon_bounds: tuple[int, int] = (0, 0)

    


@struct.dataclass
class ModelConfig:
    delta: float
    phi: float
    rho: float
    Lambda: float
    experience: float
    beta_A1: float
    beta_A2: float
    beta_A3: float
    beta_A4: float
    beta_A5: float
    beta_A6: float
    epsilon: float
    initial_attractions: jnp.array

    @staticmethod
    def create(
        key,
        priors: PriorConfig,
        game: GameConfig,
        model_bounds: dict = None,
        **overrides,
    ):
        model_bounds = model_bounds or {}
        subkeys = random.split(key, 13)

        delta_lower, delta_upper = model_bounds.get("delta", priors.delta_bounds)
        phi_lower, phi_upper = model_bounds.get("phi", priors.phi_bounds)
        rho_lower, rho_upper = model_bounds.get("rho", priors.rho_bounds)
        lambda_lower, lambda_upper = model_bounds.get("Lambda", priors.lambda_bounds)
        exp_lower, exp_upper = model_bounds.get("experience", priors.experience_bounds)
        beta_A1_lower, beta_A1_upper = model_bounds.get(
            "beta", priors.beta_A1_bounds
        )
        beta_A2_lower, beta_A2_upper = model_bounds.get(
            "beta", priors.beta_A2_bounds
        )
        beta_A3_lower, beta_A3_upper = model_bounds.get(
            "beta", priors.beta_A3_bounds
        )
        beta_A4_lower, beta_A4_upper = model_bounds.get(
            "beta", priors.beta_A4_bounds
        )
        beta_A5_lower, beta_A5_upper = model_bounds.get(
            "beta", priors.beta_A5_bounds
        )
        beta_A6_lower, beta_A6_upper = model_bounds.get(
            "beta", priors.beta_A6_bounds
        )

        epsilon_lower, epsilon_upper = model_bounds.get(
            "epsilon", priors.epsilon_bounds
        )
        # init_attractions_lower, init_attractions_upper = model_bounds.get(
        #     "initial_attractions", priors.initial_attraction_bounds
        # )

        delta = random.uniform(subkeys[0], (), minval=delta_lower, maxval=delta_upper)
        phi = random.uniform(subkeys[1], (), minval=phi_lower, maxval=phi_upper)
        rho = random.uniform(subkeys[2], (), minval=rho_lower, maxval=rho_upper)
        Lambda = random.uniform(
            subkeys[3], (), minval=lambda_lower, maxval=lambda_upper
        )
        experience = random.uniform(subkeys[4], (), minval=exp_lower, maxval=exp_upper)
        beta_A1 = random.uniform(
            subkeys[5], (), minval=beta_A1_lower, maxval=beta_A1_upper
        )
        beta_A2 = random.uniform(
            subkeys[6], (), minval=beta_A2_lower, maxval=beta_A2_upper
        )
        beta_A3 = random.uniform(
            subkeys[7], (), minval=beta_A3_lower, maxval=beta_A3_upper
        )
        beta_A4 = random.uniform(
            subkeys[8], (), minval=beta_A4_lower, maxval=beta_A4_upper
        )
        beta_A5 = random.uniform(
            subkeys[9], (), minval=beta_A5_lower, maxval=beta_A5_upper
        )
        beta_A6 = random.uniform(
            subkeys[10], (), minval=beta_A6_lower, maxval=beta_A6_upper
        )
        epsilon = random.uniform(
            subkeys[11], (), minval=epsilon_lower, maxval=epsilon_upper
        )
        # initial_attractions = random.uniform(
        #     subkeys[10],
        #     game.num_attacker_actions,
        #     minval=init_attractions_lower,
        #     maxval=init_attractions_upper,
        # )

        K = game.num_attacker_actions

        alpha = jnp.broadcast_to(priors.initial_attraction_alpha, (K,))
        beta  = jnp.broadcast_to(priors.initial_attraction_beta,  (K,))
        lo    = jnp.broadcast_to(priors.initial_attraction_lower, (K,))
        hi    = jnp.broadcast_to(priors.initial_attraction_upper, (K,))

        u = random.beta(subkeys[12], alpha, beta, shape=(K,))
        initial_attractions = lo + (hi - lo) * u

        return ModelConfig(
            delta=delta,
            phi=phi,
            rho=rho,
            Lambda=Lambda,
            experience=experience,
            beta_A1=beta_A1,
            beta_A2=beta_A2,
            beta_A3=beta_A3,
            beta_A4=beta_A4,
            beta_A5=beta_A5,
            beta_A6=beta_A6,
            epsilon=epsilon,
            initial_attractions=initial_attractions,
        ).replace(**overrides)


@struct.dataclass
class FilterConfig:
    ADP_samples: int = 200_000
    num_particles: int = NUM_PARTICLES

    @staticmethod
    def create(**kwargs):
        num_particles = int(kwargs.get("num_particles", NUM_PARTICLES))
        return FilterConfig(num_particles=num_particles)

@struct.dataclass
class MainConfig:
    game: GameConfig
    priors: PriorConfig
    model: ModelConfig
    filter: FilterConfig

    @staticmethod
    def create(
        key,
        *,
        game_kwargs=None,
        prior_kwargs=None,
        filter_kwargs=None,
        model_kwargs=None,
    ):
        game_kwargs = game_kwargs or {}
        prior_kwargs = prior_kwargs or {}
        filter_kwargs = filter_kwargs or {}
        model_kwargs = model_kwargs or {}

        subkeys = random.split(key, 2)

        game_cfg = GameConfig.create(subkeys[0], **game_kwargs)
        prior_cfg = PriorConfig(**prior_kwargs)
        model_cfg = ModelConfig.create(subkeys[1], prior_cfg, game_cfg, **model_kwargs)
        filter_cfg = FilterConfig(**filter_kwargs)

        return MainConfig(
            game=game_cfg, priors=prior_cfg, model=model_cfg, filter=filter_cfg
        )
    


def print_main_config(cfg, indent: int = 2):
    """Nicely print a MainConfig object with all nested configs."""
    
    def fmt_value(v):
        """Format floats, arrays, tuples, etc. for clean display."""
        if isinstance(v, float):
            return f"{v:.4f}"
        elif isinstance(v, (tuple, list)):
            return "(" + ", ".join(fmt_value(x) for x in v) + ")"
        elif isinstance(v, jnp.ndarray):
            # Limit length for readability
            arr = jnp.round(v, 4)
            if arr.size > 6:
                return f"array({arr[:6].tolist()} ...)"
            return f"array({arr.tolist()})"
        else:
            return str(v)

    def print_section(name, section_dict):
        print(f"\n{name}:")
        for k, v in section_dict.items():
            print(" " * indent + f"{k:<25}: {fmt_value(v)}")

    print("=" * 60)
    print("MAIN CONFIGURATION")
    print("=" * 60)

    print_section("GameConfig", vars(cfg.game))
    print_section("PriorConfig", vars(cfg.priors))
    print_section("ModelConfig", vars(cfg.model))
    print_section("FilterConfig", vars(cfg.filter))

    print("=" * 60)
    print("END CONFIG")
    print("=" * 60)


if __name__ =='__main__':

    subkey = random.PRNGKey(33)

    cfg = MainConfig.create(
        subkey
    )

    print_main_config(cfg)
