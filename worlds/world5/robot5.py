import pymunk
import numpy as np


class Robot:
    def __init__(self, mass=2, pos_init=(150, 100), length_init=150, stiffness_init=100):
        
        # constants
        self.MASS = mass
        self.RADIUS = 25
        self.EE_MASS = 0.2*self.MASS
        self.EE_RADIUS = 15
        self.LENGTH_LIMITS = (100,300)
        self.STIFFNESS_LIMITS = (50,400)
        
        # Robot's pymunk Body
        # position and velocity can be accessed by this body instance
        inertia = pymunk.moment_for_circle(self.MASS, 0, self.RADIUS)
        self.body = pymunk.Body(self.MASS, inertia)
        self.body.position = pos_init 
        self.body_shape = pymunk.Circle(self.body, self.RADIUS)
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        
        # Robot's EE's Pymunk Body
        # EE's initial mass is 0.5 of the robot's mass
        ee_inertia = pymunk.moment_for_circle(self.EE_MASS, 0, self.EE_RADIUS)
        self.EE = pymunk.Body(self.EE_MASS, ee_inertia)
        self.EE.position = (self.body.position[0], self.body.position[1] + length_init)
        self.EE_shape = pymunk.Circle(self.EE, 15)
        self.EE_contact = 0
        self.EE_previous_velocity = 0
        self.EE_acceleration = 0
        
        # Robot's Actuator Spring's Pymunk Spring
        # length and impedance can be accessed and modified by this spring instance
        self.actuator = pymunk.DampedSpring(self.body, self.EE, (0, 0), (0, 0), length_init, stiffness_init, 0)
                 
    def do_action(self, action, noise=False, l_control='direct', k_control='direct'):
        action = np.array(action)

        # constants
        alpha = 30
        l_max = self.LENGTH_LIMITS[1]
        l_min = self.LENGTH_LIMITS[0]
        l_increment_multiplier = (l_max-l_min)/alpha
        k_max = self.STIFFNESS_LIMITS[1]
        k_min = self.STIFFNESS_LIMITS[0]
        k_increment_multiplier = (k_max-k_min)/alpha
        
        # add noise to the action   
        if noise:
            action += np.random.normal(0, noise, action.shape)
        
        if l_control == 'direct':
            rest_length = action[0]*(l_max - l_min)/2 + (l_max + l_min)/2
        elif l_control == 'incremental':
            rest_length = self.actuator.rest_length + l_increment_multiplier*action[0]
            rest_length = np.clip(rest_length, l_min, l_max)
            
        if k_control == 'direct':
            stiffness = action[1]*(k_max - k_min)/2 + (k_max + k_min)/2
        elif k_control == 'incremental':
            stiffness = self.actuator.stiffness + k_increment_multiplier*action[1]
            stiffness = np.clip(stiffness, k_min, k_max)
        
        self.actuator.rest_length = rest_length
        self.actuator.stiffness = stiffness
        
        
    def reset(self, mass=2, pos_init=(150, 100), length_init=150, impedance_init=100):
        
        self.body.mass = mass
        self.body.position = pos_init
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        
        self.EE.mass = 0.2*mass
        self.EE.position = (self.body.position[0], self.body.position[1] + length_init)
        self.EE_previous_velocity = 0
        self.EE_acceleration = 0
        
        self.actuator.rest_length = length_init
        self.actuator.stiffness = impedance_init
        
    def update_accelerations(self):
        
        body_velocity_change = self.body.velocity.y - self.body_previous_velocity
        self.body_acceleration = body_velocity_change
        
        EE_velocity_change = self.EE.velocity.y - self.EE_previous_velocity
        self.EE_acceleration = EE_velocity_change
        
        self.body_previous_velocity = self.body.velocity.y
        self.EE_previous_velocity = self.EE.velocity.y