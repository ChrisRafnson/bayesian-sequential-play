"""
models.py

Python module for running a variety of opponent models under the Experience-Weighted Attraction (EWA) framework.
The goal is to be able to use these to actually run and play against EWA Opponents
"""

import ewa_update_module as ewa
import numpy as np
from config import MainConfig
from utility_functions import attacker_utility

def adjusted_sigmoid(x, beta):
    return 1 / (1 + np.exp(-(x - beta)))


class DroneAttacker:
    """
    This class represents an instance of an opponent, modeled using EWA.
    """

    def __init__(self, config: MainConfig):

        # Initialize record keeping arrays
        self.config = config
        self.num_attacker_actions = config.game.num_attacker_actions
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

        self.beta = self.config.model.beta

        #Epsilon for deviation from EWA (with probability epsilon deviate from EWA)
        self.epsilon = self.config.model.epsilon

    def select_action(self):
        """
        Select an action based upon the current attraction values.
        With probability epsilon, select a random action.

        Args:
            Self: Class function

        Returns:
            action (int): Index-form of the chosen action
        """

        sample = np.random.random()

        if sample < self.epsilon:
            #Sample random action
            action = np.random.choice(len(self.probabilities))
            self.action_record.append(action) #Store the action in index form
            return action
        else:
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
            self.action_record.append(action) #Store the action in index form
            return action

    def update_model(self, defender_action, timestep):
        """
        This method executes a single timestep in the attacker model.

        Args:
            self: The model
            defender_action (int): Action chosen by the defender

        Returns: None
        """

        true_attacker_action = self.action_record[-1]
        delta = self.config.model.delta
        phi = self.config.model.phi
        rho = self.config.model.rho
        Lambda = self.config.model.Lambda

        new_experience = rho * self.experience + 1

        new_attractions = np.zeros(self.num_attacker_actions)

        for a in range(self.num_attacker_actions):
            utility = attacker_utility(a, defender_action, self.beta, self.config, timestep)

            new_attractions[a] = (
                (phi * self.experience * self.attractions[a])
                + (
                    delta
                    + (1 - delta)
                    * ewa.indicator_function(a, true_attacker_action)
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
