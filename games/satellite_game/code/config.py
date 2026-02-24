import jax.numpy as jnp
import jax.random as random
from flax import (  # Using the flax.struct.dataclass for compatibility with jax.jit
    struct,
)


@struct.dataclass
class GameConfig:

    error_Z_variance: float  # Unique for each game instance
    initial_latent_state: (
        float  # Unique for each game instance, common across replicates
    )

    num_defender_actions: int = 3
    num_attacker_actions: int = 3

    # We multiply the matrix by some constant to allow more drastic changes in position based upon player input
    action_matrix: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array(
            [[-1.0, -0.5, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.5, 1.0]]
        )
    )

    error_X_variance: float = 1
    error_Y_variance: float = 3

    # Bounds for the starting value of the latent state
    lower_bound: int = 0
    upper_bound: int = 100

    num_timesteps: int = 100

    # The payoff parameters for the defender
    defender_beta_MSD: float = 20.0
    defender_beta_MAD: float = 50.0
    defender_beta_C: float = 100.0
    defender_beta_penalty: float = 0.1

    @staticmethod
    def create(key, **kwargs):

        subkey1, subkey2 = random.split(key)

        # Parameters for drawing the error_Z_variance
        IG_alpha: int = 2
        IG_beta: int = 1
        error_Z_variance = IG_beta / random.gamma(subkey1, IG_alpha)

        # Paramters for drawing the intial state
        lower_bound = 0
        upper_bound = 100

        initial_state = random.uniform(subkey2, minval=lower_bound, maxval=upper_bound)
        return GameConfig(
            error_Z_variance=error_Z_variance,
            initial_latent_state=initial_state,
            **kwargs,
        )


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
    lambda_bounds: tuple[int, int] = (1, 3)

    experience_parameters: tuple[float, float] = (1, 1)
    experience_bounds: tuple[int, int] = (1, 5)

    initial_attraction_parameters: tuple[float, float] = (1, 1)
    initial_attraction_bounds: tuple[int, int] = (0, 5)

    beta_MSD_parameters: tuple[float, float] = (1, 1)
    beta_MSD_bounds: tuple[int, int] = (0, 100)

    beta_MAD_parameters: tuple[float, float] = (1, 1)
    beta_MAD_bounds: tuple[int, int] = (0, 100)

    beta_C_parameters: tuple[float, float] = (1, 1)
    beta_C_bounds: tuple[int, int] = (0, 100)

    beta_penalty_parameters: tuple[float, float] = (1, 1)
    beta_penalty_bounds: tuple[int, int] = (0, 1)

    latent_state_parameters: tuple[float, float] = (1, 1)
    latent_state_bounds: tuple[int, int] = (0, 100)

    eta_parameters: tuple[int, int] = (2, 1)


@struct.dataclass
class ModelConfig:
    delta: float
    phi: float
    rho: float
    Lambda: float
    experience: float
    beta_MSD: float
    beta_MAD: float
    beta_C: float
    beta_penalty: float
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
        subkeys = random.split(key, 10)

        delta_lower, delta_upper = model_bounds.get("delta", priors.delta_bounds)
        phi_lower, phi_upper = model_bounds.get("phi", priors.phi_bounds)
        rho_lower, rho_upper = model_bounds.get("rho", priors.rho_bounds)
        lambda_lower, lambda_upper = model_bounds.get("Lambda", priors.lambda_bounds)
        exp_lower, exp_upper = model_bounds.get("experience", priors.experience_bounds)
        beta_MSD_lower, beta_MSD_upper = model_bounds.get(
            "beta_MSD", priors.beta_MSD_bounds
        )
        beta_MAD_lower, beta_MAD_upper = model_bounds.get(
            "beta_MAD", priors.beta_MAD_bounds
        )
        beta_C_lower, beta_C_upper = model_bounds.get("beta_C", priors.beta_C_bounds)
        beta_penalty_lower, beta_penalty_upper = model_bounds.get(
            "beta_penalty", priors.beta_penalty_bounds
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
        beta_MSD = random.uniform(
            subkeys[5], (), minval=beta_MSD_lower, maxval=beta_MSD_upper
        )
        beta_MAD = random.uniform(
            subkeys[6], (), minval=beta_MAD_lower, maxval=beta_MAD_upper
        )
        beta_C = random.uniform(
            subkeys[7], (), minval=beta_C_lower, maxval=beta_C_upper
        )
        beta_penalty = random.uniform(
            subkeys[8], (), minval=beta_penalty_lower, maxval=beta_penalty_upper
        )
        initial_attractions = random.uniform(
            subkeys[9],
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
            beta_MSD=beta_MSD,
            beta_MAD=beta_MAD,
            beta_C=beta_C,
            beta_penalty=beta_penalty,
            initial_attractions=initial_attractions,
        ).replace(**overrides)


@struct.dataclass
class FilterConfig:
    num_particles: int = 10_000


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
