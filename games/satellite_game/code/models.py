"""
models.py

Python module for running a variety of opponent models under the Experience-Weighted Attraction (EWA) framework.
The goal is to be able to use these to actually run and play against EWA Opponents
"""

import ewa_update_module as ewa
import numpy as np
from config import MainConfig
from utility_functions import get_utility


def adjusted_sigmoid(x, beta):
    return 1 / (1 + np.exp(-(x - beta)))


class SatelliteOperator:
    """
    This class represents an instance of an opponent, modeled using EWA.
    """

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

        # log the initial attractions
        self.attractions = self.config.model.initial_attractions
        self.attraction_record.append(self.attractions.copy())

        # convert initial attractions in probabilities
        self.probabilities = ewa.softmax_probabilities(
            self.attractions, self.config.model.Lambda
        )

        # log initial probabilities
        self.probability_record.append(self.probabilities.copy())

        # log initial experience
        self.experience = self.config.model.experience
        self.experience_record.append(self.experience)

        self.betas = [
            self.config.model.beta_MSD,
            self.config.model.beta_MAD,
            self.config.model.beta_C,
            self.config.model.beta_penalty,
        ]

    def utility_function(self, observed_state, action):
        """
        A function to return the utility for the actions taken in an observed state.
        The game being played is the 1-dimensional version of the proximity game.

        Args:
            observed_state (float): The observed distance from the other player (can be either attacker's or defender's observation)
            betas (array): The payoff parameters beta_1, beta_2, beta_3

        Returns:
            (float): The utility for the given actions and observation state
        """

        return get_utility(self.betas, observed_state, action)

    def select_action(self):
        """
        Select an action based upon the current attraction values
        """

        # Recalculate safe probabilities (defensive, even if already stored)
        probs = np.clip(self.probabilities, 0.0, 1.0)
        total = np.sum(probs)
        if total == 0.0 or np.isnan(total):
            # Fall back to uniform if all-zero or invalid
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / total

        # Now safely sample
        action = np.random.choice(len(probs), p=probs)
        self.action_record.append(action)
        return action

    def update_model(self, defender_action, attacker_observation):
        """
        This method executes a single timestep in the state space model. It accepts the friendly action and then
        selects an action for the opponent according to the EWA model. It then plays the game, records data about the
        game, and updates the opponent's EWA.

        Returns: action played by the opponent and the payoff
        """

        opponent_action = self.action_record[-1]
        delta = self.config.model.delta
        phi = self.config.model.phi
        rho = self.config.model.rho
        Lambda = self.config.model.Lambda
        action_matrix = self.config.game.action_matrix

        new_experience = rho * self.experience + 1

        new_attractions = np.zeros(3)

        for a in range(3):
            K_t = (
                attacker_observation
                - action_matrix[self.action_record[-1], defender_action]
                + action_matrix[a, defender_action]
            )
            utility = self.utility_function(K_t, a)

            new_attractions[a] = (
                (phi * self.experience * self.attractions[a])
                + (
                    delta
                    + (1 - delta)
                    * ewa.indicator_function(a, self.action_record[-1])
                    * utility
                )
            ) / new_experience

        new_probabilities = ewa.softmax_probabilities(new_attractions, Lambda)

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
