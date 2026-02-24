"""
benchmark_policies.py

Contains the implementation of the benchmark polcies (Myopic Clairvoyant and Fictitious Play).
"""

import ewa_update_module as ewa
import numpy as np
import jax
import jax.numpy as jnp
from config import MainConfig
from utility_functions import defender_utility

class FP_policy:
    def __init__(self, config: MainConfig):

        # Initialize record keeping arrays
        self.config = config
        self.num_actions = config.game.num_defender_actions
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

        # log initial experience
        self.experience = 0
        self.experience_record.append(self.experience)

        # log the initial attractions
        self.attractions = np.ones(self.num_actions)
        self.attraction_record.append(self.attractions.copy())

        # convert initial attractions in probabilities
        self.probabilities = ewa.softmax_probabilities(self.attractions, self.Lambda)

        # log initial probabilities
        self.probability_record.append(self.probabilities.copy())

    def select_action(self):
        """
        Select an action based upon the current attraction values
        """

        action = np.random.choice(np.arange(self.config.game.num_defender_actions), p=self.probabilities)
        self.action_record.append(action)
        return action

    def update_model(self, attacker_action):
        """
        This method executes a single timestep in the state space model. It accepts the friendly action and then
        selects an action for the opponent according to the EWA model. It then plays the game, records data about the
        game, and updates the opponent's EWA.

        Returns: action played by the opponent and the payoff
        """

        defender_action = self.action_record[-1]

        new_experience = self.rho * self.experience + 1

        new_attractions = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            utility = defender_utility(attacker_action, defender_action, self.config)

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


def get_MC_action(probabilities, config: MainConfig):
    num_def_actions = config.game.num_defender_actions
    num_atk_actions = config.game.num_attacker_actions
    def_actions = np.arange(1, num_def_actions+1)
    atk_actions = np.arange(1, num_atk_actions+1)

    expected_returns = [0 for i in range(num_def_actions)]

    for defender_action in def_actions:  # Loop across all friendly actions
        for attacker_action in atk_actions:  # Loop across all opponent actions
            expected_returns[defender_action-1] += probabilities[attacker_action-1] * defender_utility(attacker_action, defender_action, config)
    return int(np.argmax(expected_returns)) #Add plus one to bring the index from (0, 9) -> (1, 10)

def get_MC_action_jax(probabilities, config: MainConfig, timestep, prev_attacker_action=-1, prev_defender_action=-1):
    num_def_actions = config.game.num_defender_actions
    num_atk_actions = config.game.num_attacker_actions

    # def defender_return(defender_action):
    #     # Vectorize over attacker actions
    #     atk_actions = jnp.arange(num_atk_actions)
    #     utils = jax.vmap(lambda atk: defender_utility(atk, defender_action, config, timestep=timestep))(atk_actions)
    #     return jnp.sum(probabilities * utils)
    
    def defender_return(defender_action):
        # Vectorize over attacker actions
        atk_actions = jnp.arange(num_atk_actions)
        utils = jax.vmap(defender_utility, in_axes=(0, None, None, None, None, None))(
            atk_actions, defender_action, config,
            timestep, prev_attacker_action, prev_defender_action
        )
        return jnp.sum(probabilities * utils)


    # Vectorize over defender actions
    def_actions = jnp.arange(num_def_actions)
    expected_returns = jax.vmap(defender_return)(def_actions)

    return jnp.argmax(expected_returns)

class DirichletMultinomial:
    def __init__(self, tao, config: MainConfig):

        # Initialize record keeping arrays
        self.config = config
        self.num_def_actions = config.game.num_defender_actions
        self.num_atk_actions = config.game.num_attacker_actions
        self.action_record = []
        self.def_actions = jnp.arange(self.num_def_actions)

        self.tao = tao
        self.belief_state = jnp.ones(self.num_atk_actions) #Flat prior for the dirichlet
        self.timestep = 0

    def select_action(self, prev_attacker_action=-1, prev_defender_action=-1):
        """
        Select an action based upon the current model values
        """

        probabilities = self.belief_state / jnp.sum(self.belief_state) #Calculate the new probabilities based on the current beliefs
        num_def_actions = self.num_def_actions
        num_atk_actions = self.num_atk_actions
        config = self.config
        timestep = self.timestep

        # def defender_return(defender_action):
        #     # Vectorize over attacker actions
        #     atk_actions = jnp.arange(num_atk_actions)
        #     utils = jax.vmap(lambda atk: defender_utility(atk, defender_action, config, self.timestep))(atk_actions)
        #     return jnp.sum(probabilities * utils)
        
        def defender_return(defender_action):
            # Vectorize over attacker actions
            atk_actions = jnp.arange(num_atk_actions)
            utils = jax.vmap(defender_utility, in_axes=(0, None, None, None, None, None))(
                atk_actions, defender_action, config,
                timestep, prev_attacker_action, prev_defender_action
            )
            return jnp.sum(probabilities * utils)
        
        # Vectorize over defender actions
        def_actions = self.def_actions
        expected_returns = jax.vmap(defender_return)(def_actions)

        #Select the action with the largest expected return
        action = jnp.argmax(expected_returns)
        self.action_record.append(action)
        return action

    def update_model(self, attacker_action):
        """
        Applies the conjugacy update given an attacker's action.
        """

        #grab the previous belief state
        belief_state = self.belief_state

        #Increment the parameter associated with the attacker's action by one
        belief_state = belief_state.at[attacker_action].add(1)
        
        #Divide by the tao parameter and assign the new belief state to the model
        self.belief_state = belief_state / self.tao

        self.timestep += 1



