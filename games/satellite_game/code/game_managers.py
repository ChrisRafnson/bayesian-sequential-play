import numpy.random as random
from config import MainConfig


class SatelliteOperationGame:
    """
    A class to manage the execution of the Satellite Operations Game.
    """

    def __init__(self, config: MainConfig):
        self.action_matrix = config.game.action_matrix
        self.error_X_sigma = config.game.error_X_variance
        self.error_Y_sigma = config.game.error_Y_variance
        self.error_Z_sigma = config.game.error_Z_variance
        self.state_record = []

        self.state = config.game.initial_latent_state
        self.state_record.append(self.state)

    def step(self, attacker_action, defender_action):

        transition = self.state + self.action_matrix[attacker_action, defender_action]

        error_X = random.normal(0, self.error_X_sigma)  # Sample random noise for X
        error_Y = random.normal(0, self.error_Y_sigma)  # Error for the defender
        error_Z = random.normal(0, self.error_Z_sigma)  # Error for the attacker

        self.state = transition + error_X  # Save the new latent state
        self.state_record.append(self.state)  # record the new latent state

        attacker_observation = self.state + error_Z
        defender_observation = self.state + error_Y

        return (
            self.state,
            defender_observation,
            attacker_observation,
        )  # Returns the defender and attacker sensor measurements, respectively
