"""
models.py

Python module for running a variety of opponent models under the Experience-Weighted Attraction (EWA) framework.
The goal is to be able to use these to actually run and play against EWA Opponents
"""

import ewa_update_module as ewa
import numpy as np
from config import MainConfig
from utility_functions import utility
from utils import kinematic_transition, inverse_kinematic_transition

def adjusted_sigmoid(x, beta):
    return 1 / (1 + np.exp(-(x - beta)))


class MannedVehicle:
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

        self.beta_A1 = self.config.model.beta_A1
        self.beta_A2 = self.config.model.beta_A2
        self.beta_A3 = self.config.model.beta_A3
        self.beta_A4 = self.config.model.beta_A4
        self.beta_A5 = self.config.model.beta_A5
        self.beta_A6 = self.config.model.beta_A6

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

    def update_model(self, attacker_observation, defender_action, config):
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

            defender_inputs = config.game.defender_action_set[defender_action]

            XD1_pre, XD2_pre, XD3_pre, XD4_pre = inverse_kinematic_transition(
                attacker_observation[0],
                attacker_observation[1],
                attacker_observation[2],
                attacker_observation[3],
                config.game.L,
                config.game.delta_t,
                defender_inputs[0],
                defender_inputs[1]
            )

            attacker_inputs = config.game.attacker_action_set[true_attacker_action]

            XA1_pre, XA2_pre, XA3_pre, XA4_pre = inverse_kinematic_transition(
                attacker_observation[4],
                attacker_observation[5],
                attacker_observation[6],
                attacker_observation[7],
                config.game.L,
                config.game.delta_t,
                attacker_inputs[0],
                attacker_inputs[1]
            )

            #Extract the acceleration and steering angle associated with the current attraction being updated
            new_attacker_inputs = config.game.attacker_action_set[a]

            #Using the new action for the attacker, we convert to a theoretical post-decision state
            #Note that when the attraction being updated is the same as the true action take, our calculated
            #post-decision state will be the same as the attacker's observation

            XD1_post, XD2_post, XD3_post, XD4_post = kinematic_transition(
                XD1_pre,
                XD2_pre,
                XD3_pre,
                XD4_pre,
                config.game.L,
                config.game.delta_t,
                defender_inputs[0],
                defender_inputs[1]
            )

            XA1_post, XA2_post, XA3_post, XA4_post = kinematic_transition(
                XA1_pre,
                XA2_pre,
                XA3_pre,
                XA4_pre,
                config.game.L,
                config.game.delta_t,
                new_attacker_inputs[0],
                new_attacker_inputs[1]
            )

            #Now that we have a post-decision state for our current attraction, we can calculate the utility we
            #would receive
            attacker_utility = utility(
                XA2_post, #The "true" vertical position of the attacker
                XA1_post, #Attacker horizontal
                XA2_post, #Attacker vertical
                XA3_post, #Attacker heading
                XA4_post, #Attacker speed
                XD1_post, #Attacker's opponent's horizontal A.K.A defender horizontal
                XD2_post, #defender vertical
                XD3_post, #defender heading
                XD4_post, #defender speed
                a,
                self.beta_A1,
                self.beta_A2,
                self.beta_A3,
                self.beta_A4,
                self.beta_A5,
                self.beta_A6,
                config,
                is_defender=False
    )

            new_attractions[a] = (
                (phi * self.experience * self.attractions[a])
                + (
                    delta
                    + (1 - delta)
                    * ewa.indicator_function(a, true_attacker_action)
                    * attacker_utility
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
