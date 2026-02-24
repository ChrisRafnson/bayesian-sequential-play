"""
benchmark_policies.py

Contains the implementation of the benchmark polcies (Myopic Clairvoyant and Fictitious Play).
"""

import ewa_update_module as ewa
import numpy as np
from config import MainConfig
from utility_functions import get_utility

class FP_policy:
    def __init__(self, config: MainConfig):

        # Initialize record keeping arrays
        self.config = config
        self.action_record = (
            []
        )  # Store tuples at each time step, storing the actions taken by the model
        self.attraction_record = (
            []
        )  # Store attractions at each time step for modeled player
        self.probability_record = (
            []
        )  # Store probabilities of play at each time step for modeled player
        self.experience_record = (
            []
        )  # Store experieince at each time step for modeled player
        self.payoff_record = []

        # Set up parameters
        self.delta = 1
        self.phi = 1
        self.rho = 1
        self.Lambda = 2

        self.defender_betas = np.array(
            [
                config.game.defender_beta_MSD,
                config.game.defender_beta_MAD,
                config.game.defender_beta_C,
                config.game.defender_beta_penalty,
            ]
        )

        # log initial experience
        self.experience = 0
        self.experience_record.append(self.experience)

        # log the initial attractions
        self.attractions = np.ones(config.game.num_defender_actions)
        self.attraction_record.append(self.attractions.copy())

        # convert initial attractions in probabilities
        self.probabilities = ewa.softmax_probabilities(self.attractions, self.Lambda)

        # log initial probabilities
        self.probability_record.append(self.probabilities.copy())

    def select_action(self):
        """
        Select an action based upon the current attraction values
        """

        action = np.random.choice(len(self.probabilities), p=self.probabilities)
        self.action_record.append(action)
        return action

    def update_model(self, attacker_action, defender_observation):
        """
        This method executes a single timestep in the state space model. It accepts the friendly action and then
        selects an action for the opponent according to the EWA model. It then plays the game, records data about the
        game, and updates the opponent's EWA.

        Returns: action played by the opponent and the payoff
        """

        defender_action = self.action_record[-1]
        action_matrix = self.config.game.action_matrix

        new_experience = self.rho * self.experience + 1

        new_attractions = np.zeros(3)

        for a in range(3):
            K_t = (
                defender_observation
                - action_matrix[attacker_action, self.action_record[-1]]
                + action_matrix[attacker_action, a]
            )
            utility = get_utility(self.defender_betas, K_t, a)

            new_attractions[a] = (
                (self.phi * self.experience * self.attractions[a])
                + (
                    self.delta
                    + (1 - self.delta)
                    * ewa.indicator_function(self.action_record[-1], a)
                    * utility
                )
            ) / new_experience

        new_probabilities = ewa.softmax_probabilities(new_attractions, self.Lambda)

        self.probabilities = new_probabilities
        self.attractions = new_attractions
        self.experience = new_experience
        self.probability_record.append(new_probabilities.copy())
        self.experience_record.append(new_experience)
        self.attraction_record.append(new_attractions.copy())

    def combine_data(self):

        # Combine the experience, attractions, and probabilities in that order
        experience = np.array(self.experience_record).reshape(-1, 1)
        attractions = np.array(self.attraction_record)
        probabilities = np.array(self.probability_record)

        joined = np.hstack([experience, attractions, probabilities])

        parameters = np.array(list(self.parameters.values()))
        parameters_tiled = np.tile(parameters, (joined.shape[0], 1))

        joined = np.hstack([parameters_tiled, joined])

        return joined[
            :-1, :
        ]  # We drop the last row because of the extra EWA calculation at the end.


def get_MC_action(model, manager, config: MainConfig):
    probabilities = model.probabilities
    num_actions = len(probabilities)
    state = manager.state
    defender_betas = np.array(
        [
            config.game.defender_beta_MSD,
            config.game.defender_beta_MAD,
            config.game.defender_beta_C,
            config.game.defender_beta_penalty,
        ]
    )

    expected_returns = [0 for i in range(len(probabilities))]

    for i in range(num_actions):  # Loop across all friendly actions
        for j in range(num_actions):  # Loop across all opponent actions
            transition = state + config.game.action_matrix[tuple([j, i])]
            expected_returns[i] += probabilities[j] * get_utility(
                defender_betas, transition, i
            )

    return int(np.argmax(expected_returns))
