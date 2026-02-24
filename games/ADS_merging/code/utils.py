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

def kinematic_transition(X1, X2, X3, X4, L, delta_t, acceleration, steering_angle):
    """
    A function to perform kinematic updates for either the defender or attacker or latent state

    Args:
        X1 (float): The horizontal position in meters
        X2 (float): The vertical position in meters
        X3 (float): The heading angle in degrees
        X4 (float): The speed in meters/second
        L (float): The interaxle distance
        delta_t (float): The size of the discretized time intervals
        acceleration (float): The ADS or MV acceleration input
        steering_angle (float): The ADS or MV steering input


    Return:
        X1 (float): The updated horizontal position in meters
        X2 (float): The updated vertical position in meters
        X3 (float): The updated heading angle in degrees
        X4 (float): The updated speed in meters/second
    """

    # # --- INPUTS ---
    # jax.debug.print(
    #     "KIN IN: X1={x1}, X2={x2}, X3(deg)={x3}, X4={x4}, a={a}, steer(deg)={s}",
    #     x1=X1, x2=X2, x3=X3, x4=X4, a=acceleration, s=steering_angle,
    #     ordered=True,
    # )

    #Convert to radians
    X3 = jnp.deg2rad(X3)
    steering_angle = jnp.deg2rad(steering_angle)

    # #Calculate the first derivatives of each state variable
    # X1_dx = X4 * jnp.sin(X3)
    # X2_dx = X4 * jnp.cos(X3)
    # X3_dx = (X4/L) * jnp.tan(steering_angle)
    # X4_dx = acceleration

    #Alternative derivatives, where the steering angle input for timestep t+1 affects the velocity vectors of t+1 instead of t+2
    #Calculate the first derivatives of each state variable
    X3_dx = (X4/L) * jnp.tan(steering_angle)
    X3_new = X3 + X3_dx * delta_t


    X1_dx = X4 * jnp.sin(X3_new)
    X2_dx = X4 * jnp.cos(X3_new)
    X4_dx = acceleration

    # jax.debug.print(
    #     "KIN DERIVS: dX1={dx1}, dX2={dx2}, dX3(rad)={dx3}, dX4={dx4}",
    #     dx1=X1_dx, dx2=X2_dx, dx3=X3_dx, dx4=X4_dx,
    #     ordered=True,
    # )

    #Update the state variables using their derivatives
    X1 = X1 + X1_dx * delta_t
    X2 = X2 + X2_dx * delta_t
    X4 = X4 + X4_dx * delta_t
    X4 = jnp.maximum(X4, 0.0)   # <-- prevent negative speeds

    #Convert back to degrees
    X3 = jnp.rad2deg(X3_new)

    # # --- OUTPUTS ---
    # jax.debug.print(
    #     "KIN OUT: X1={x1}, X2={x2}, X3(deg)={x3}, X4={x4}",
    #     x1=X1, x2=X2, x3=X3, x4=X4,
    #     ordered=True,
    # )

    return X1, X2, X3, X4

def inverse_kinematic_transition(X1_new, X2_new, X3_new, X4_new, L, delta_t, acceleration, steering_angle):
    """
    A function to perform recover the approximate starting kinematic state after receiving the new states

    Args:
        X1_new (float): The horizontal position in meters
        X2_new (float): The vertical position in meters
        X3_new (float): The heading angle in degrees
        X4_new (float): The speed in meters/second
        L (float): The interaxle distance
        delta_t (float): The size of the discretized time intervals
        acceleration (float): The ADS or MV acceleration input
        steering_angle (float): The ADS or MV steering input


    Return:
        X1 (float): The approximate original horizontal position in meters
        X2 (float): The approximate original vertical position in meters
        X3 (float): The approximate original heading angle in degrees
        X4 (float): The approximate original speed in meters/second
    """

    #Convert to radians
    X3_new = jnp.deg2rad(X3_new)
    steering_angle = jnp.deg2rad(steering_angle)

    # Forward: X4_new = X4 + acceleration * dt
    # Backward:
    X4 = X4_new - acceleration * delta_t

    # Forward: heading_new = heading + (X4/L)*tan(steer)*dt
    # Backward:
    heading = X3_new - (X4 / L) * jnp.tan(steering_angle) * delta_t

    # Now heading corresponds to X3 (but still in radians)

    # Forward position updates:
    #   X1_new = X1 + X4 * sin(heading) * dt
    #   X2_new = X2 + X4 * cos(heading) * dt
    # Backward:
    X1 = X1_new - X4 * jnp.sin(heading) * delta_t
    X2 = X2_new - X4 * jnp.cos(heading) * delta_t

    # Convert heading back to degrees
    X3 = jnp.rad2deg(heading)

    return X1, X2, X3, X4




