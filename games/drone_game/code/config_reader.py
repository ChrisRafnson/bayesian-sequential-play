import pickle

file_path = "/home/chris/Desktop/bayesian-security-games/games/drone_game/drone_game_instances_TEST/SEED_0_FLAT(1.0, 1.0, 1.0)i0.pkl"

with open(file_path, "rb") as f:
    config = pickle.load(f)

print(config.model)
