"""
utils.py

Contains a variety of general functions required by other components of the package.
"""

import jax.numpy as jnp

def safe_softmax(x, axis=-1, k=1.0, eps=1e-12):
    x = jnp.array(x) * k  # Apply Lambda scaling here
    x = x - jnp.max(x, axis=axis, keepdims=True)  # for numerical stability
    exps = jnp.exp(x)
    sum_exps = jnp.sum(exps, axis=axis, keepdims=True)
    sum_exps = jnp.where(sum_exps > 0, sum_exps, eps)  # avoid div by zero
    probs = exps / sum_exps
    probs = jnp.clip(probs, 0.0, 1.0)  # remove tiny negatives
    return probs / jnp.sum(probs, axis=axis, keepdims=True)  # final normalize
