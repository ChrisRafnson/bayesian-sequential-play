"""
regressions.py

Regression functions and helpers for the Approximate Dynamic Programming policy.
"""

import jax.numpy as jnp

def OLS_regression(X, y):
    thetas, residuals, rank, s = jnp.linalg.lstsq(X, y)
    return thetas

def combine_features(*arrays, add_bias=False):
    """
    Combines multiple feature arrays into a single design matrix.

    Args:
        *arrays: one or more (N, d) feature arrays to concatenate.
        add_bias (bool): if True, prepends a column of ones for intercept.

    Returns:
        X_combined: (N, sum of d_i [+1 if bias]) array
    """
    X = jnp.hstack(arrays)
    if add_bias:
        bias = jnp.ones((X.shape[0], 1))
        X = jnp.hstack([bias, X])
    return X

def predict_linear(X, thetas):
    regression_results = X @ thetas[:18]
    return regression_results.reshape(-1, 1)  # We reshape so that it is (N, 1)


def XGB_regression(X, y):
    pass
