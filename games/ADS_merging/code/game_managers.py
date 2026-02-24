import numpy.random as random
from config import MainConfig
from utils import kinematic_transition
import jax.numpy as jnp


class ADSGameManager:
    """
    A class to manage the execution of the Satellite Operations Game.
    """

    def __init__(self, config: MainConfig):

        #Action Sets
        self.defender_action_set = config.game.defender_action_set
        self.attacker_action_set = config.game.attacker_action_set

        #Noise Variances
        self.error_X_sigma = config.game.error_X_sigma
        self.error_Y_sigma = config.game.error_Y_sigma
        self.error_Z_sigma = config.game.error_Z_sigma

        #Game Constants
        self.delta_t = config.game.delta_t
        self.L = config.game.L #Interaxle distance
        self.l1 = config.game.l1 #The outer boundary of the left lane
        self.l2 = config.game.l2 #The centerline
        self.l3 = config.game.l3 #The outer boundary of the right lane

        self.W1 = config.game.W1 #The soft merge point
        self.W2 = config.game.W2 #The hard merge point, only one lane hereafter

        #Initial Game State
        self.XD1 = config.game.XD1_init #Starting value for XD1 (meters)
        self.XD2 = config.game.XD2_init #Starting value for XD2 (meters)
        self.XD3 = config.game.XD3_init #Starting value for XD3 (degrees)
        self.XD4 = config.game.XD4_init #Starting value for XD4 (m/s)

        self.XA1 = config.game.XA1_init #Starting value for XA1 (meters)
        self.XA2 = config.game.XA2_init #Starting value for XA2 (meters)
        self.XA3 = config.game.XA3_init #Starting value for XA3 (dsegrees)
        self.XA4 = config.game.XA4_init #Starting value for XA4 (m/s)

        #Package state variables and record values
        self.state_record = []

        self.state = jnp.array([self.XD1, self.XD2, self.XD3, self.XD4, self.XA1, self.XA2, self.XA3, self.XA4])
        self.state_record.append(self.state)

    def step(self, attacker_action, defender_action):

        def_input = self.defender_action_set[defender_action]
        atk_input = self.attacker_action_set[attacker_action]


        #Transition the defenders kinematics first
        XD1, XD2, XD3, XD4 = kinematic_transition(
            self.XD1,
            self.XD2,
            self.XD3,
            self.XD4,
            self.L,
            self.delta_t,
            def_input[0],
            def_input[1]
        )

        #Transition the attackers kinematics next
        XA1, XA2, XA3, XA4 = kinematic_transition(
            self.XA1,
            self.XA2,
            self.XA3,
            self.XA4,
            self.L,
            self.delta_t,
            atk_input[0],
            atk_input[1]
        )

        #Construct an array of the latent state vars
        state = jnp.array([XD1, XD2, XD3, XD4, XA1, XA2, XA3, XA4])

        #Sample Latent State Error

        noise = random.normal(0.0, self.error_X_sigma)

        state = jnp.asarray(state).reshape(-1)
        noise = jnp.asarray(noise).reshape(-1)


        XD1, XD2, XD3, XD4, XA1, XA2, XA3, XA4 = state + noise

        #Prevent speed from going negative
        XD4 = max(XD4, 0.0)
        XA4 = max(XA4, 0.0) 


        self.XD1 = XD1
        self.XD2 = XD2
        self.XD3 = XD3
        self.XD4 = XD4
        self.XA1 = XA1
        self.XA2 = XA2
        self.XA3 = XA3
        self.XA4 = XA4

        # #Sample latent state error
        # XD1 = XD1 + random.normal(0, self.error_X_sigma)
        # XD2 = XD2 + random.normal(0, self.error_X_sigma)
        # XD3 = XD3 + random.normal(0, self.error_X_sigma)
        # XD4 = XD4 + random.normal(0, self.error_X_sigma)
        # XD4 = jnp.maximum(XD4, 0.0)   # <-- prevent negative speeds

        # #Sample latent state error
        # XA1 = XA1 + random.normal(0, self.error_X_sigma)
        # XA2 = XA2 + random.normal(0, self.error_X_sigma)
        # XA3 = XA3 + random.normal(0, self.error_X_sigma)
        # XA4 = XA4 + random.normal(0, self.error_X_sigma)
        # XA4 = jnp.maximum(XA4, 0.0)   # <-- prevent negative speeds

        #Package the defender state variables and add noise
        defender_observation_noise = random.normal(0, self.error_Y_sigma)
        defender_observation = jnp.array([XD1, XD2, XD3, XD4, XA1, XA2, XA3, XA4]) + defender_observation_noise

        #Package the attacker state variables and add noise
        attacker_observation_noise = random.normal(0, self.error_Z_sigma)
        attacker_observation = jnp.array([XD1, XD2, XD3, XD4, XA1, XA2, XA3, XA4]) + attacker_observation_noise

        #Package both states and record the values
        self.state = jnp.array([self.XD1, self.XD2, self.XD3, self.XD4, self.XA1, self.XA2, self.XA3, self.XA4])
        self.state_record.append(self.state)

        return defender_observation, attacker_observation
