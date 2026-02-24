import time

import filters as jf
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotting as plotting
from benchmark_policies import FP_policy, get_MC_action_jax, DirichletMultinomial
from config import MainConfig
from models import DroneAttacker
from policies import (
    generate_all_trajectories,
    get_boltzmann_action,
    get_H2S_action,
    get_means,
)
from utility_functions import defender_utility, attacker_utility
from ADP import get_ADP_action_MLP, compute_value_functions
import regressions as rf


def evaluate_ADP_policy(key, seed, config: MainConfig, model_path):

    np.random.seed(seed)  # Set seed for reproduceability

    # Initialize simulation components
    num_particles = config.filter.num_particles
    num_attacker_actions = config.game.num_attacker_actions
    num_defender_actions = config.game.num_defender_actions
    num_timesteps = config.game.num_timesteps

    key, subkey = jax.random.split(key)

    model = DroneAttacker(config)
    particles = jf.initialize_particles_beta_prior(
        subkey, config, num_particles, num_attacker_actions
    )

    # Initialize record keeping arrays
    particle_means_record = []
    action_record = []
    utility_record = []
    policy_compute_time_record = []

    filter_means_record = []
    filter_stds_record = []
    filter_p5_record = []
    filter_p95_record = []

    defender_actions = jnp.arange(num_defender_actions)

    model_seq = rf.load_params_sequence(model_path, num_timesteps)

    # Play the merge game
    for timestep in range(num_timesteps):

        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        min_particle = jf.get_mins(particles)
        max_particle = jf.get_maxs(particles)

        #If allowing a decision every epoch use this block and comment out the next.
        key, subkey = jax.random.split(key)
        start_time = time.perf_counter()
        defender_action = get_ADP_action_MLP(mean_particle, model_seq[timestep], defender_actions, config)
        end_time = time.perf_counter()

        # # Determine defender action
        # if timestep % 2 == 0:
        #     key, subkey = jax.random.split(key)
        #     start_time = time.perf_counter()
        #     defender_action = get_ADP_action_MLP(mean_particle, model_seq[timestep], defender_actions, config)
        #     end_time = time.perf_counter()
        #     prev_atk_act = -1
        #     prev_def_act = -1

        #Get attacker actions
        attacker_action = model.select_action()

        policy_compute_time_record.append(end_time - start_time)

        # Store the actions for each player
        actions = [attacker_action, defender_action]
        action_record.append(actions)

        # The opponent's model is then updated
        model.update_model(defender_action, timestep)

        # Store the utilities given to each player
        atk_util = attacker_utility(
            attacker_action, defender_action, model.beta, config,timestep=timestep
        )
        def_util = defender_utility(
            attacker_action, defender_action, config,
            timestep=timestep,
            prev_attacker_action=prev_atk_act,
            prev_defender_action=prev_def_act
        )
        utilities = [atk_util, def_util]
        utility_record.append(utilities)

        prev_atk_act = attacker_action
        prev_def_act = defender_action

        # Finally, we update our particle filter
        key, subkey = jax.random.split(key)
        particles = jf.step(
            subkey,
            particles,
            attacker_action,
            defender_action,
            config,
        )

        # Store particle stats from the updated particles
        mean_particle = get_means(particles)
        std_particle = jf.get_stds(particles)
        p5_particle = np.percentile(np.array(particles), 5, axis=0)
        p95_particle = np.percentile(np.array(particles), 95, axis=0)

        particle_means_record.append(mean_particle)
        filter_means_record.append(mean_particle)
        filter_stds_record.append(std_particle)
        filter_p5_record.append(p5_particle)
        filter_p95_record.append(p95_particle)

        results = {
            "policy": f"ADP",
            "particle_means": particle_means_record,
            "filter_means": filter_means_record,
            "filter_stds": filter_stds_record,
            "filter_p5": filter_p5_record,
            "filter_p95": filter_p95_record,
            "actions": action_record,
            "utilities": utility_record,
            "compute_times": policy_compute_time_record,
            "opponent_prob_record": model.probability_record,
            "attacker_betas": model.beta,
        }
    
    return results

def compute_episode_summary(results: dict, config: MainConfig):

    utility_record = results["utilities"]
    policy_compute_time_record = results["compute_times"]
    action_record = np.array(results["actions"])

    # ----------------------------Utility Related Statistics-----------------------------------------

    # Average and Total Util

    UR = np.array(utility_record)
    attacker_utilities = UR[:, 0]
    defender_utilities = UR[:, 1]

    atk_avg_util = np.average(attacker_utilities)
    def_avg_util = np.average(defender_utilities)

    atk_tot_util = np.sum(attacker_utilities)
    def_tot_util = np.sum(defender_utilities)

    # Window-based Util
    window_size = 10
    UR = np.array(results["utilities"])  # shape (T,2)
    df = pd.DataFrame(UR, columns=["attacker", "defender"])
    df["step"] = np.arange(len(df))
    df["window_id"] = df["step"] // window_size
    win_df = df.groupby("window_id")[["attacker", "defender"]].mean().reset_index()
    win_df["advantage"] = win_df["defender"] - win_df["attacker"]

    # ----------------------------Action Related Statistics-----------------------------------------

    # --- Action distributions ---
    T = action_record.shape[0]
    attacker_actions = action_record[:, 0]
    defender_actions = action_record[:, 1]

    if config is not None:
        atk_counts = np.bincount(
            attacker_actions, minlength=config.game.num_attacker_actions
        )
        def_counts = np.bincount(
            defender_actions, minlength=config.game.num_defender_actions
        )
    else:
        atk_counts = np.bincount(attacker_actions)
        def_counts = np.bincount(defender_actions)

    atk_dist = (atk_counts / T).tolist()
    def_dist = (def_counts / T).tolist()

    # # joint distribution
    # max_a, max_d = atk_counts.size, def_counts.size
    # joint_counts = np.zeros((max_a, max_d), dtype=int)
    # for a, d in action_record:
    #     joint_counts[a, d] += 1
    # joint_dist = (joint_counts / T).tolist()

    # ----------------------------Computing Time Related Statistics-----------------------------------------
    PCTR = np.array(policy_compute_time_record)
    avg_comp_time = np.average(PCTR)

    episode_summary = {
        "policy": str(results["policy"]),
        "avg_comp_time": avg_comp_time,
        "atk_avg_util": atk_avg_util,
        "atk_tot_util": atk_tot_util,
        "def_avg_util": def_avg_util,
        "def_tot_util": def_tot_util,
        "atk_action_dist": atk_dist,
        "def_action_dist": def_dist,
        "attacker_actions": attacker_actions.tolist(),
        "defender_actions": defender_actions.tolist(),
        "attacker_utilities": attacker_utilities.tolist(),
        "defender_utilities": defender_utilities.tolist(),
    }

    return episode_summary


def print_episode_summary(episode_summary: dict, withHeader=True):
    policy_name = episode_summary["policy"]
    avg_comp_time = episode_summary["avg_comp_time"]
    atk_avg_util = episode_summary["atk_avg_util"]
    atk_tot_util = episode_summary["atk_tot_util"]
    def_avg_util = episode_summary["def_avg_util"]
    def_tot_util = episode_summary["def_tot_util"]

    # Define column headers and widths
    headers = [
        "Policy",
        "Avg Comp Time",
        "Atk Avg Util",
        "Atk Tot Util",
        "Def Avg Util",
        "Def Tot Util",
    ]
    values = [
        policy_name,
        avg_comp_time,
        atk_avg_util,
        atk_tot_util,
        def_avg_util,
        def_tot_util,
    ]
    widths = [15, 15, 15, 15, 15, 15]

    # Format values: if numeric → 4 decimals
    def fmt(val):
        if isinstance(val, (int, float)):
            return f"{val:.4f}"
        return str(val)

    formatted_values = [fmt(v) for v in values]

    if withHeader:
        # Print header
        header_row = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
        print(header_row)
        print("-" * len(header_row))

    # Print values
    value_row = "".join(f"{v:<{w}}" for v, w in zip(formatted_values, widths))
    print(value_row)


if __name__ == "__main__":

    key = jax.random.key(42)
    seed = 42

    key, subkey = jax.random.split(key)
    config = MainConfig.create(subkey)

    H = 0

    trajectories = generate_all_trajectories(H, config)

    # results = evaluate_boltzmann_policy(key, seed, 2, config)
    # results = evaluate_H2S_policy(key, seed, config, H, trajectories=trajectories)
    # results = evaluate_AMG_policy(key, seed, config)
    results = evaluate_MC_policy(key, seed, config)
    results = evaluate_FP_policy(key, seed, config)

    # key, subkey = jax.random.split(key)
    # start = time.perf_counter()
    # model_matrix = compute_value_functions(subkey, config, config.game.num_timesteps)
    # end = time.perf_counter()
    # print(f"Time to compute value functions: {end - start:.4f} seconds")
    # results = evaluate_ADP_policy(key, seed, config, model_matrix)

    episode_summary = compute_episode_summary(results, config)
    print_episode_summary(episode_summary)