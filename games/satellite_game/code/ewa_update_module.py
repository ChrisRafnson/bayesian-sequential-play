import time
from math import exp

import jax.numpy as jnp
import numpy as np
from jax.nn import softmax


def indicator_function(x, y):
    """
    This function implements a simple indicator, return 1 if the inputs are the same and 0 if they aren not.

    Inputs:

        x: some input
        y: some input

    Outputs:

        1 if x == y, 0 otherwise
    """

    if x == y:
        return 1
    else:
        return 0


def softmax_probabilities(attractions, Lambda):
    z = jnp.array(attractions) * Lambda
    p = softmax(z, axis=-1)

    # explicitly normalize & fix rounding errors
    p = jnp.clip(p, 0, 1)  # ensure no negatives
    p /= jnp.sum(p)  # normalize
    p = p.at[-1].set(1.0 - jnp.sum(p[:-1]))  # force last entry to absorb residual
    p = np.array(p)
    return p


def first_nonzero_index(arr):
    return int(np.argmax(arr != 0)) if np.any(arr != 0) else -1


def generate_payoff_dictionary(utility_matrix, action_labels):
    """
    This function generates a structure-compliant payoff dictionary to be used within the EWA update.

    Inputs:

        utility_matrix: a multidimensional array containing the rewards for a combination of opponent and friendly actions
        action_labels: the labels associated with each index in the matrix. It is assumed that the action in the 0th index of
            a column or row in the payoff matrix is associated with the 0th index of the action labels. Otherwise the actions
            will not be labeled properly

    Outputs:

        payoff_dictionary: a nested dictionary containing the payoffs for a combination of actions.
            The keys are pulled from the action labels. The first index of the dictionary is the opponent action
            while the second index is associated with the friendly action. Example usage is payoff_dictionary[opponent_action][friendly_action].
    """

    payoff_dictionary = dict()

    for action in action_labels:
        payoff_dictionary[action] = dict()

    for row_index in range(len(utility_matrix)):
        for col_index in range(len(utility_matrix)):

            payoff_dictionary[action_labels[row_index]][action_labels[col_index]] = (
                utility_matrix[row_index][col_index]
            )

    return payoff_dictionary


def update_EWA(parameters, attractions, experience, actions, payoff_tensor):
    """
    This function implements a full update of the Experience weighted attraction model, as defined by Camerer and Ho in their
    paper "Experience-Weighted Attraction Learning in Normal Form Games".

    Inputs:

        parameters: a dictionary containing the EWA parameters delta, phi, rho, and lambda
        attractions: a vector containing the attractions from the current time step (should have the same length as the first dimension of the payoff tensor)
        experience: the experience weight of the current time step
        actions: a vector actions taken by each player
        payoff_tensor: n-dimensional tensor containing the payoff information for each combination of actions among players (number of dimensions should be the same as the action vector).
            This function assumes that the first dimension of the tensor is associated with the actions taken by same player for this EWA model
        probability form: a string to specify the function to transform attractions into probabilities. May be 'logit' or 'power', will default to power if there is an invalid input

    Outputs:

        probabilities: a vecotr containing probability of play for each action, according to the updated EWA model of the opponent
        new_attractions: a vector of the new action attractions, according to the updated EWA model
        new_experience: the new experience weight, according to the updated EWA model
    """
    num_attractions = len(attractions)  # Number of attractions in the attraction vector
    num_actions = payoff_tensor.shape[
        0
    ]  # Number of actions in the payoff tensor for the player we are modeling
    action_taken = actions[0]  # Gets the action for our modeled player

    new_experience = 0
    new_attractions = np.zeros(num_actions)
    probabilities = np.zeros(num_actions)

    # The simplest update to the EWA model involves updating the experience weighting using the rho parameter
    new_experience = (parameters["rho"] * experience) + 1

    # Following an update to the attraction weighting, we now proceed to update the actual attractions of each action
    for i in range(num_attractions):
        new_attractions[i] = (
            (parameters["phi"] * experience * attractions[i])
            + (
                parameters["delta"]
                + (1 - parameters["delta"]) * indicator_function(i, action_taken)
            )
            * payoff_tensor[tuple([i] + actions[1:])]
        ) / new_experience

    # Softmax Probability
    # running_total = 0
    # for i in range(num_attractions):
    #     #We only calculate the numerator of the logit function as we require the summation accross all actions for the denominator.
    #     probabilities[i] = exp(parameters['lambda'] * new_attractions[i])
    #     running_total += probabilities[i]

    probabilities = softmax(np.array(attractions) * parameters["lambda"])

    # Now we normalize to get valid probabilities
    # probabilities = probabilities / running_total

    # Old code for alternative probability form
    #     running_total = 0
    #     for action in new_attractions:
    #         #We only calculate the numerator of the logit function as we require the summation accross all actions for the denominator.
    #         probabilities[action] = new_attractions[action] ** parameters['lambda']
    #         running_total += probabilities[action]
    #     #Now we normalize to get valid probabilities
    #     probabilities = {k: v / running_total for k, v in probabilities.items()}

    return probabilities, new_attractions, new_experience


def update_opponent_EWA(
    parameters,
    attractions,
    experience,
    opponent_action,
    friendly_action,
    payoff_dictionary,
    probability_form,
):
    """
    This function implements a full update of the Experience weighted attraction model, as defined by Camerer and Ho in their
    paper "Experience-Weighted Attraction Learning in Normal Form Games".

    Inputs:

        parameters: a dictionary containing the EWA parameters delta, phi, rho, and lambda
        attractions: a dictionary containing the opponents attractions from the current time step
        experience: the experience weight of the current time step
        opponent_action: the action taken by the opponent in the current time step
        friendly_action: the action taken by the blue side
        payoff_matrix: the game payoffs, stored in this case as a nested dictionary, the first index is the action taken by the opponent and the second is the action taken by the blue side
        probability form: a string to specify the function to transform attractions into probabilities. May be 'logit' or 'power', will default to power if there is an invalid input

    Outputs:

        probabilities: a dictionary probability of play for each action, according to the updated EWA model of the opponent
        new_attractions: a dictionary of the new action attractions, according to the updated EWA model
        new_experience: a dictionary of the new experience weight, according to the updated EWA model
    """

    new_experience = 0
    new_attractions = dict()
    probabilities = dict()

    # The simplest update to the EWA model involves updating the experience weighting using the rho parameter
    new_experience = (parameters["rho"] * experience) + 1

    # Following an update to the attraction weighting, we now proceed to update the actual attractions of each action
    for action in attractions:
        new_attractions[action] = (
            (parameters["phi"] * experience * attractions[action])
            + (
                parameters["delta"]
                + (1 - parameters["delta"])
                * indicator_function(action, opponent_action)
            )
            * payoff_dictionary[action][friendly_action]
        ) / new_experience

    if probability_form == "logit":
        running_total = 0
        for action in new_attractions:
            # We only calculate the numerator of the logit function as we require the summation accross all actions for the denominator.
            probabilities[action] = exp(parameters["lambda"] * new_attractions[action])
            running_total += probabilities[action]

        # Now we normalize to get valid probabilities
        probabilities = {k: v / running_total for k, v in probabilities.items()}

    else:
        running_total = 0
        for action in new_attractions:
            # We only calculate the numerator of the logit function as we require the summation accross all actions for the denominator.
            probabilities[action] = new_attractions[action] ** parameters["lambda"]
            running_total += probabilities[action]
        # Now we normalize to get valid probabilities
        probabilities = {k: v / running_total for k, v in probabilities.items()}
    return probabilities, new_attractions, new_experience


def print_opponent_stats(i, probabilities, attractions, experience):
    """
    A simple function to print the opponents statistics at a specific time step

    Inputs:

        i: the current timestep
        probabilities: a dictionary containing the probabilities of each action being played according the EWA model
        attractions: a dictionary containing the attractions of the opponent to each action
        experience: the value of the experience weighting at the current time step

    Outputs to console.
    """

    print(f"\n======== Time Step {i + 1} ========")
    print("Opponent Action Probabilities:")
    for action, prob in probabilities.items():
        print(f"  {action:<15}: {prob:.10f}")

    print("Opponent Action Attractions:")
    for action, attr in attractions.items():
        print(f"  {action:<15}: {attr:.10f}")

    print(f"\nExperience Weight: {experience:.10f}")


if __name__ == "__main__":

    # # The following code serves as an example of how one might use the EWA model, more importantly it provides an example of proper inputs into the model.
    # # I assume that the game being played is a matrix game or could be formulated in such a way.

    # action_labels = ['swerve', 'straight'] # Definition of the action space for each opponent

    # #The payoff matrix for the enemy, the blue force does not require a payoff matrix unless we are observing both sides
    # enemy_utility_matrix = [[0, -1],
    #                         [1, -100]]

    # #The parameters for the EWA model
    # parameters = {
    #     'delta':0.75,
    #     'rho':0.75,
    #     'phi':0.75,
    #     'lambda':1
    # }

    # #The attraction to each action and the experience weight must be defined before running the model as well
    # experience = 4
    # attractions = {
    #     'swerve':1,
    #     'straight':1
    # }

    # #Create a payoff_dictionary, we could've just used the index of the payoff matrix but the dictionary allows us to use the actual action names
    # payoff_dictionary = generate_payoff_dictionary(enemy_utility_matrix, action_labels)

    # opponent_action = action_labels[np.random.binomial(1, p=0.5)] #Draw a random action for the initial opponent move
    # friendly_action = action_labels[np.random.binomial(1, p=0.5)] #Draw a random action for the initial friendly move

    # for i in range(25):

    #     #Update the EWA model
    #     probabilities, attractions, experience = update_opponent_EWA(
    #         parameters,
    #         attractions,
    #         experience,
    #         opponent_action,
    #         friendly_action,
    #         payoff_dictionary,
    #         probability_form='logit'
    #     )

    #     print(f"Opponent Move: {opponent_action}")
    #     print(f"Friendly Move: {friendly_action}")
    #     print_opponent_stats(i, probabilities, attractions, experience)

    #     friendly_action = action_labels[np.random.binomial(1, p=0.5)] #We will use a random policy just for simplicity
    #     opponent_action = action_labels[np.random.binomial(1, p=probabilities['straight'])] # The opponents action is generated via a bernoulli trial with p = probability of choosing the action 'straight'

    # TEST OF MULTIDIMENSIONAL EWA UPDATE

    # Define number of players, this defines the number of axes for our tensor/hypercube
    num_players = 2

    # Define number of actions per player, assuming each player has the same number of actions. This forces a hypercube shape
    num_actions = 10

    # Dynamically define tensor shape
    shape = tuple(np.ones(num_players, dtype=int) * num_actions)
    print(shape)

    # Randomly draw a hypercube filled with integers on the [0, 10] range
    np.random.seed(42)
    payoff_tensor = np.random.randint(low=0, high=10, size=shape)
    print(payoff_tensor)

    for idx, val in np.ndenumerate(payoff_tensor):
        print(f"{idx}: {val}")

    # The parameters for the EWA model
    parameters = {"delta": 0.75, "rho": 0.75, "phi": 0.75, "lambda": 1}
    experience = 6

    # Now we define our initial attractions, this is done dynamically
    attractions = [np.random.randint(0, 5) for i in range(num_actions)]
    print(f"Initial Attractions: {attractions}")

    # Define initial actions taken by the players
    actions = [
        first_nonzero_index(
            np.random.multinomial(n=1, pvals=[1 / num_actions] * num_actions, size=1)
        )
        for i in range(num_players)
    ]
    print(f"Initial Actions: {actions}")

    for i in range(25):

        start = time.time()
        # Update the EWA model
        probabilities, attractions, experience = update_EWA(
            parameters, attractions, experience, actions, payoff_tensor
        )
        delta = time.time() - start

        print(f"\nTimestep {i}:")
        print(f"Time for EWA Update: {delta}")
        print(f"Modeled Player Move: {actions[0]}")
        print(f"Other Moves: {actions[1:]}")
        print(f"Probabilities of Play: {probabilities}")
        print(f"Attractions:           {attractions}")
        print(experience)

        # This gets the action from the player we are modeling, according to the action probs
        modeled_action = first_nonzero_index(
            np.random.multinomial(n=1, pvals=probabilities, size=1)
        )

        # Get the actions from the other players, modeled randomly
        other_actions = [
            first_nonzero_index(
                np.random.multinomial(
                    n=1, pvals=[1 / num_actions] * num_actions, size=1
                )
            )
            for i in range(num_players - 1)
        ]

        actions = [modeled_action] + other_actions
