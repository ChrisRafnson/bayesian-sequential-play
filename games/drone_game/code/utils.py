"""
utils.py

Contains a variety of general functions required by other components of the package.
"""

import jax.numpy as jnp
import jax


def safe_softmax(x, axis=-1, k=1.0, eps=1e-12):
    x = jnp.array(x) * k  # Apply Lambda scaling here
    x = x - jnp.max(x, axis=axis, keepdims=True)  # for numerical stability
    exps = jnp.exp(x)
    sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
    sum_exps = jnp.where(sum_exps > 0, sum_exps, eps)  # avoid div by zero
    probs = exps / sum_exps
    probs = jnp.clip(probs, 0.0, 1.0)  # remove tiny negatives
    return probs / jnp.sum(probs, axis=axis, keepdims=True)  # final normalize

def ij_to_index(i, j, n=10):

    """
    A function to map an array [i,j] to its index k, using row major order.

    Args:
        i (int): The first entry of the array. Corresponds to the first site chosen by the attacker.
        j (int): The second entry of the array Corresponds to the second site chosen by the attacker.
        n (int): The number of sites

    Outputs:
        k (int): The unique index k. Corresponds to the action index of [i, j]
    """

    return (i) * n + (j)

@jax.jit
def index_to_ij(k, n=10):

    """
    A function to map an action k to its array [i, j], using row major order.

    Args:
        k (int): The unique index k. Corresponds to the action index of [i, j]
        n (int): The number of sites

    Outputs:
        action_array (array): An array containing the following:
            i (int): The first entry of the array. Corresponds to the first site chosen by the attacker.
            j (int): The second entry of the array Corresponds to the second site chosen by the attacker.
    """

    i = k // n
    j = k % n
    return jnp.stack([i, j], axis = -1)

def random_swap(key, arr):
    """Randomly swap two distinct elements of a JAX array."""
    n = arr.shape[0]
    i, j = jax.random.randint(key, (2,), 0, n)
    
    # Ensure distinct indices by resampling j if needed
    j = jax.lax.cond(i == j,
                     lambda _: (j + 1) % n,  # simple deterministic fix if equal
                     lambda _: j,
                     operand=None)
    
    # Perform the swap without Python side effects
    arr_swapped = arr.at[i].set(arr[j])
    arr_swapped = arr_swapped.at[j].set(arr[i])
    return arr_swapped

def random_neighbor_swap(key, arr):
    """Randomly swap neighboring elements in a JAX array."""
    n = arr.shape[0]
    # Pick a random index i, and swap i with i+1
    i = jax.random.randint(key, (), 0, n - 1)
    arr_swapped = arr.at[i].set(arr[i + 1])
    arr_swapped = arr_swapped.at[i + 1].set(arr[i])
    return arr_swapped


