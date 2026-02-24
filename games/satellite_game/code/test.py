import csv
import hashlib
import os
import pickle
import time
from pathlib import Path
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from config import MainConfig


CONFIG_DIR = Path(
    "/home/chris/Desktop/bayesian-security-games/games/satellite_game/satellite_game_instances_new"
)

config_paths = sorted(CONFIG_DIR.glob("*.pkl"))

starting_record = []

# Begin evaluations
for config_path in config_paths:
    config_name = config_path.stem
    print(f"Evaluating config: {config_name}")

    with config_path.open("rb") as f:
        config = pickle.load(f)

        starting_record.append(config.game.initial_latent_state)

# --- Analyze distribution of initial states ---
starting_array = np.array(starting_record)

print("Summary statistics of initial states:")
print(f"Count: {starting_array.size}")
print(f"Mean: {starting_array.mean():.4f}")
print(f"Std:  {starting_array.std():.4f}")
print(f"Min:  {starting_array.min():.4f}")
print(f"Max:  {starting_array.max():.4f}")

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(starting_array, bins=30, color="steelblue", edgecolor="black", alpha=0.7)
plt.title("Distribution of Initial States")
plt.xlabel("Initial State Value")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.4)
plt.savefig("starting_state_dist.png")