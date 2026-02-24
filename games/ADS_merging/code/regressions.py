"""
Regression functions and helpers for the Approximate Dynamic Programming policy.
Includes both analytical (OLS) and neural (Flax MLP) regression models.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax import struct
import optax
import flax.serialization
import msgpack
import os
from constants import HUBER_DELTA

# ------------------------------------------------------------
# Existing OLS section (unchanged)
# ------------------------------------------------------------
from constants import MAX_PARTICLE_DIM, LOGICAL_PARTICLE_DIM


def OLS_regression(X, y):
    thetas, residuals, rank, s = jnp.linalg.lstsq(X, y)
    return thetas


def combine_features(*arrays, add_bias=False):
    """
    Combines multiple feature arrays into a single design matrix.
    """
    X = jnp.hstack(arrays)
    if add_bias:
        bias = jnp.ones((X.shape[0], 1))
        X = jnp.hstack([bias, X])
    return X


def predict_linear(X, thetas):
    regression_results = X @ thetas[:LOGICAL_PARTICLE_DIM + 1]
    return regression_results.reshape(-1, 1)


def XGB_regression(X, y):
    pass


# ------------------------------------------------------------
# Neural network regression (Flax MLP) section
# ------------------------------------------------------------
# class MLPRegressor(nn.Module):
#     """Simple feed-forward MLP for regression."""
#     hidden_dims: list[int]
#     output_dim: int = 1

#     @nn.compact
#     def __call__(self, x):
#         for h in self.hidden_dims:
#             x = nn.relu(nn.Dense(h)(x))
#         return nn.Dense(self.output_dim)(x)

def huber_loss(preds, targets, delta=HUBER_DELTA):
    diff = preds - targets
    return jnp.mean(
        jnp.where(
            jnp.abs(diff) < delta,
            0.5 * diff**2,
            delta * (jnp.abs(diff) - 0.5 * delta)
        )
    )


class MLPRegressor(nn.Module):
    hidden_dims: list[int]
    activation: str = "relu"
    output_dim: int = 1

    def _activation(self, x):
        if self.activation == "relu":
            return nn.relu(x)
        elif self.activation == "tanh":
            return nn.tanh(x)
        elif self.activation == "sigmoid":
            return nn.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = self._activation(x)
        return nn.Dense(self.output_dim)(x)


@struct.dataclass
class TrainState(train_state.TrainState):
    X_mean: jnp.ndarray = None
    X_std:  jnp.ndarray = None
    y_mean: jnp.ndarray = None
    y_std:  jnp.ndarray = None
    is_terminal: bool = struct.field(pytree_node=False, default=False)




# def create_train_state(rng, model, input_dim, lr=1e-3):
#     """Initialize optimizer + parameters."""
#     params = model.init(rng, jnp.ones((1, input_dim)))
#     tx = optax.adam(lr)
#     return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# def create_train_state(rng, model, input_dim, lr=1e-3):
#     params = model.init(rng, jnp.ones((1, input_dim)))
#     tx = optax.adam(lr)
#     return TrainState.create(
#         apply_fn=model.apply,
#         params=params,
#         tx=tx,
#         X_mean=jnp.zeros((1, input_dim)),
#         X_std=jnp.ones((1, input_dim)),
#         y_mean=jnp.zeros((1, 1)),
#         y_std=jnp.ones((1, 1)),
#     )

def create_train_state(rng, model, input_dim, lr=1e-3):
    params = model.init(rng, jnp.ones((1, input_dim)))
    tx = optax.adam(lr)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        X_mean=jnp.zeros((1, input_dim)),
        X_std=jnp.ones((1, input_dim)),
        y_mean=jnp.zeros((1, 1)),
        y_std=jnp.ones((1, 1)),
        is_terminal=False,
    )

@jax.jit
def train_step(state, x, y):
    """Single MSE gradient update."""
    def loss_fn(params):
        preds = state.apply_fn(params, x)

        # return jnp.mean((preds - y) ** 2)

        diff = preds - y
        loss = jnp.mean(jnp.where(jnp.abs(diff) < HUBER_DELTA,
                          0.5 * diff**2,
                          1.0 * (jnp.abs(diff) - 0.5)))
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def predict_mlp(model, params, x):
    """Apply trained MLP."""
    return model.apply(params, x)


# ------------------------------------------------------------
# Multi-stage ADP support
# ------------------------------------------------------------
# def initialize_mlp_sequence(num_stages, input_dim, hidden_dims, rng, lr=1e-3):
#     """Initialize a list of MLP TrainStates (one per stage)."""
#     model = MLPRegressor(hidden_dims=hidden_dims)
#     rngs = jax.random.split(rng, num_stages)
#     return [create_train_state(r, model, input_dim, lr) for r in rngs]

def initialize_mlp_sequence(
    num_stages,
    input_dim,
    hidden_dims,
    rng,
    lr=1e-3,
    activation="relu",
):
    model = MLPRegressor(
        hidden_dims=hidden_dims,
        activation=activation,
    )
    rngs = jax.random.split(rng, num_stages)
    return [
        create_train_state(r, model, input_dim, lr)
        for r in rngs
    ]



def apply_mlp_sequence(model, params_sequence, x, t):
    """Apply stage-t MLP."""
    return model.apply(params_sequence[t], x)


# ------------------------------------------------------------
# Save / load utilities
# ------------------------------------------------------------
def save_params_sequence(params_sequence, filepath):
    """Save all model parameters in one file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = flax.serialization.to_bytes(params_sequence)
    with open(filepath, "wb") as f:
        f.write(data)


def load_params_sequence(filepath, num_stages):
    """Load model parameters from a msgpack file."""
    with open(filepath, "rb") as f:
        data = f.read()
    dummy_structure = [None] * num_stages
    return flax.serialization.from_bytes(dummy_structure, data)

# def load_state_sequence(filepath, num_stages, input_dim, hidden_dims, rng, lr=0.0):
#     with open(filepath, "rb") as f:
#         data = f.read()

#     dummy_seq = initialize_mlp_sequence(
#         num_stages=num_stages,
#         input_dim=input_dim,
#         hidden_dims=hidden_dims,
#         rng=rng,
#         lr=lr,
#     )
#     return flax.serialization.from_bytes(dummy_seq, data)

def load_state_sequence(
    filepath,
    num_stages,
    input_dim,
    hidden_dims,
    rng,
    lr=0.0,
):
    with open(filepath, "rb") as f:
        data = f.read()

    model_seq = initialize_mlp_sequence(
        num_stages=num_stages,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        rng=rng,
        lr=lr,
    )

    model_seq = flax.serialization.from_bytes(model_seq, data)

    # 🔴 CRITICAL: reapply terminal flag
    final_t = num_stages - 1
    model_seq[final_t] = model_seq[final_t].replace(is_terminal=True)

    return model_seq


def restore_terminal_flags(model_seq):
    final_t = len(model_seq) - 1
    model_seq[final_t] = model_seq[final_t].replace(is_terminal=True)
    return model_seq



if __name__ == "__main__":

    #Example usage for MLP

    rng = jax.random.PRNGKey(0)
    seq = initialize_mlp_sequence(num_stages=10, input_dim=8, hidden_dims=[64, 32], rng=rng)

    # training example
    X = jax.random.normal(rng, (100, 8))
    y = jnp.sum(X, axis=1, keepdims=True)
    seq[0] = train_step(seq[0], X, y)

    # saving / loading
    save_params_sequence([s.params for s in seq], "checkpoints/mlp_sequence.msgpack")
    loaded = load_params_sequence("checkpoints/mlp_sequence.msgpack", 10)

    # prediction
    model = MLPRegressor(hidden_dims=[64, 32])
    y_pred = model.apply(loaded[0], X[:1]).squeeze()

    print(y_pred)
