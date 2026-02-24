import jax
import jax.numpy as jnp
import jax.random as random


def draw_beta_AB(key, shape, alpha, beta, lower_bound, upper_bound):
    # Sample the beta distribution
    samples = jax.random.beta(key, alpha, beta, shape)

    # Convert the samples back to the specified range.
    samples = samples * (upper_bound - lower_bound) + lower_bound
    return samples


def draw_beta(key, shape, mean=0.5, scale=2.0, lower_bound=0, upper_bound=1):
    """
    An adjusted version of the jax.random.beta method, using mean and scale paramters. Additionally,
    this version supports values on a range larger than [0, 1].

    Args:
        key (PRNG_key): The key to be consumed by the function
        shape (tuple): The shape of the output
        mean (float): The mean parameter of the beta distribution. Must be between the requested range, otherwise it will be clipped slightly inward
        scale (float): The scale parameter of the beta distribution (also called the sample size). Must be greater than zero

    Returns:
        samples: The samples drawn from the beta distribution

    Notes: This function will draw beta values on a [0, 1] scale. If the mean and scale are not provided,
    then the beta is parameterized to a uniform distribution
    """

    # First we adjust the mean to the [0, 1] range

    mean = (mean - lower_bound) / (upper_bound - lower_bound)
    mean = jnp.clip(mean, 1e-3, 1 - 1e-3)  # avoid alpha=0 or beta=0
    # Calculate the alpha and beta parameters
    alpha = mean * scale
    beta = (1 - mean) * scale

    # Sample the beta distribution
    samples = jax.random.beta(key, alpha, beta, shape)

    # Convert the samples back to the specified range.
    samples = samples * (upper_bound - lower_bound) + lower_bound
    return samples

# #Not being used from what I can tell
# def format_prior(lower_bound, upper_bound, mean=None, scale=None):
#     """
#     A help function to create properly formatted tuples for the draw_beta() function. Mainly
#     allowing the user to specify a flat (uniform) prior by setting the mean and scale to None.

#     Args:
#         lower_bound (float): The lower bound of your prior belief
#         upper_bound (float): The upper bound of your prior belief
#         mean (float): The center of your prior belief, this is an optional parameter
#         scale (float): The concentration of density around the center of your belief, this is another optional parameter

#     Returns
#         prior (tuple): A properly formatted tuple to provide to the draw_beta() function
#     """

#     if mean == None or scale == None:
#         mean = (upper_bound - lower_bound) / 2 + lower_bound
#         prior = (mean, 2.0, lower_bound, upper_bound)
#     else:
#         prior = (mean, scale, lower_bound, upper_bound)
#     return prior

# #Not being used from what I can tell
# def sample_beta_triplet(key, mean, scale, lower_bound, upper_bound):
#     """We can come back to this for now we will assume three separate priors on each beta"""

#     subkey1, subkey2, subkey3 = random.split(key, 3)
#     beta_MAD = random.uniform(subkey1, minval=lower_bound, maxval=upper_bound)
#     beta_MSD = random.uniform(subkey2, minval=lower_bound, maxval=beta_MAD)
#     beta_C = random.uniform(subkey3, minval=lower_bound, maxval=upper_bound)
#     return jnp.array([beta_MSD, beta_MAD, beta_C])


def _initial_MSD_sample_given_MAD(key, alpha, beta, lower_bound, MAD):
    return draw_beta_AB(
        key, shape=(), alpha=alpha, beta=beta, lower_bound=lower_bound, upper_bound=MAD
    )


def initial_MSD_sample_given_MAD(
    key, num_samples, MAD_vals, alpha, beta, lower_bound, upper_bound
):

    keys = random.split(key, num=num_samples)

    MSD_samples = jax.vmap(
        _initial_MSD_sample_given_MAD, in_axes=(0, None, None, None, 0)
    )(keys, alpha, beta, lower_bound, MAD_vals)
    return jnp.array(MSD_samples)


def evolve_MSD_given_MAD(key, shape, scale, MSD, MAD, lower_bound):
    return draw_beta(
        key,
        shape=shape,
        mean=MSD,
        scale=scale,
        lower_bound=lower_bound,
        upper_bound=MAD,
    )
