import jax.numpy as jnp
import jax.random as random
from flax import (  # Using the flax.struct.dataclass for compatibility with jax.jit
    struct,
)
from constants import NUM_PARTICLES
import jax.numpy as jnp
from pprint import pformat
from utils import random_swap, random_neighbor_swap





@struct.dataclass
class GameConfig:
    num_sites: int = 8
    num_defender_selections: int = 1
    num_attacker_selections: int = 1
    num_defender_actions: int = None
    num_attacker_actions: int = None
    num_timesteps: int = 30

    utility_a: float = 1.0
    utility_b: float = 1.0
    utility_c: float = 1.0

    # priority_lists: jnp.array = None

    @staticmethod
    def create(**kwargs):
        # Compute derived values *before* constructing GameConfig
        num_sites = int(kwargs.get("num_sites", 8))
        num_defender_selections = int(kwargs.get("num_defender_selections", 1))
        num_attacker_selections = int(kwargs.get("num_attacker_selections", 1))

        kwargs["num_defender_actions"] = num_sites ** num_defender_selections
        kwargs["num_attacker_actions"] = num_sites ** num_attacker_selections
        kwargs["num_timesteps"] = kwargs.get("num_timesteps", 30)

        num_timesteps = kwargs["num_timesteps"]

        # priority_lists = jnp.zeros(shape=(num_timesteps, num_sites)) #This will contain the priority lists for each timestep
        # # priority_list = jnp.arange(1, num_sites+1) #This is the starting template of site priorities
        # priority_list = jnp.ones(num_sites)

        # for timestep in range(num_timesteps):
        #     key = random.PRNGKey(timestep)
        #     priority_list = random_neighbor_swap(key, priority_list) #Swaps two adjacent bases priorities 
        #     priority_lists = priority_lists.at[timestep].set(priority_list)

        utility_a = float(kwargs.get("utility_a", 1.0))
        utility_b = float(kwargs.get("utility_b", 1.0))
        utility_c = float(kwargs.get("utility_c", 1.0))

        kwargs["utility_a"] = utility_a
        kwargs["utility_b"] = utility_b
        kwargs["utility_c"] = utility_c

        # kwargs["priority_lists"] = priority_lists
        return GameConfig(**kwargs)





@struct.dataclass
class PriorConfig:

    # Note: The parameters refer to the mean and scale parameters for a beta distribution.
    # The bounds refer to the limits of the beta distribution

    AEP_scale: int = 1000

    delta_parameters: tuple[float, float] = (1, 1)
    delta_bounds: tuple[int, int] = (0, 1)

    phi_parameters: tuple[float, float] = (1, 1)
    phi_bounds: tuple[int, int] = (0, 1)

    rho_parameters: tuple[float, float] = (1, 1)
    rho_bounds: tuple[int, int] = (0, 1)

    lambda_parameters: tuple[float, float] = (1, 1)
    lambda_bounds: tuple[int, int] = (1, 3)

    experience_parameters: tuple[float, float] = (1, 1)
    experience_bounds: tuple[int, int] = (1, 5)

    initial_attraction_parameters: tuple[float, float] = (1, 1)
    initial_attraction_bounds: tuple[int, int] = (0, 5)

    beta_parameters: tuple[float, float] = (1, 1)
    beta_bounds: tuple[int, int] = (0, 5)

    epsilon_parameters: tuple[float, float] = (1, 1)
    epsilon_bounds: tuple[int, int] = (0, 1)

    


@struct.dataclass
class ModelConfig:
    delta: float
    phi: float
    rho: float
    Lambda: float
    experience: float
    beta: float
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
        subkeys = random.split(key, 8)

        delta_lower, delta_upper = model_bounds.get("delta", priors.delta_bounds)
        phi_lower, phi_upper = model_bounds.get("phi", priors.phi_bounds)
        rho_lower, rho_upper = model_bounds.get("rho", priors.rho_bounds)
        lambda_lower, lambda_upper = model_bounds.get("Lambda", priors.lambda_bounds)
        exp_lower, exp_upper = model_bounds.get("experience", priors.experience_bounds)
        beta_lower, beta_upper = model_bounds.get(
            "beta", priors.beta_bounds
        )
        epsilon_lower, epsilon_upper = model_bounds.get(
            "epsilon", priors.epsilon_bounds
        )
        init_attractions_lower, init_attractions_upper = model_bounds.get(
            "initial_attractions", priors.initial_attraction_bounds
        )

        delta = random.uniform(subkeys[0], (), minval=delta_lower, maxval=delta_upper)
        phi = random.uniform(subkeys[1], (), minval=phi_lower, maxval=phi_upper)
        rho = random.uniform(subkeys[2], (), minval=rho_lower, maxval=rho_upper)
        Lambda = random.uniform(
            subkeys[3], (), minval=lambda_lower, maxval=lambda_upper
        )
        experience = random.uniform(subkeys[4], (), minval=exp_lower, maxval=exp_upper)
        beta = random.uniform(
            subkeys[5], (), minval=beta_lower, maxval=beta_upper
        )
        epsilon = random.uniform(
            subkeys[6], (), minval=epsilon_lower, maxval=epsilon_upper
        )
        initial_attractions = random.uniform(
            subkeys[7],
            game.num_attacker_actions,
            minval=init_attractions_lower,
            maxval=init_attractions_upper,
        )

        return ModelConfig(
            delta=delta,
            phi=phi,
            rho=rho,
            Lambda=Lambda,
            experience=experience,
            beta=beta,
            epsilon=epsilon,
            initial_attractions=initial_attractions,
        ).replace(**overrides)


@struct.dataclass
class FilterConfig:
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

        game_cfg = GameConfig.create(**game_kwargs)
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
        subkey,
        game_kwargs={
            "num_sites": 8,
            "num_attacker_selections": 1,
            "utility_a": 5.0,
            "utility_b": 5.0,
            "utility_c": 5.0},
        model_kwargs={"epsilon": jnp.array(0.0)}
    )

    print_main_config(cfg)
