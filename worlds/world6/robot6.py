import pymunk
import numpy as np
import pymunk.pygame_util
import pygame


class Robot:
    def __init__(self, pos_init=(150, 100), length_init=100, stiffness_init=100):
        
        # constants
        self.MASS = m = 1
        self.RADIUS = r = 25
        self.EE_MASS = ee_m = self.MASS*0.2
        self.EE_RADIUS = ee_r = 15
        self.LENGTH_LIMITS = (50,300)
        self.STIFFNESS_LIMITS = (50,400)
        self.ACTUATOR_DIM_MULTIPLIER = adm = 50
        
        # Robot's pymunk Body
        # position and velocity can be accessed by this body instance
        inertia = pymunk.moment_for_circle(self.MASS, 0, self.RADIUS)
        self.body = pymunk.Body(self.MASS, inertia)
        self.body.position = pos_init 
        self.body_shape = pymunk.Circle(self.body, self.RADIUS)
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        
        # Robot's EE is now made of 2 links.
        # upper_arm is the shorter link accounting for 3/5 of total mass
        # dimensions are l = 3*adm - r - ee_r - 10 and b = 10
        upper_arm_dim = (adm*3 - r - ee_r - 10, 10)
        upper_arm_inertia = pymunk.moment_for_box(ee_m*3/5, upper_arm_dim)
        self.upper_arm = pymunk.Body(ee_m*3/5, upper_arm_inertia)
        self.upper_arm_shape = pymunk.Poly.create_box(self.upper_arm, upper_arm_dim)
        self.upper_arm_shape.color = (0, 0, 255, 255)
        self.upper_arm_shape.position = (
            self.body.position[0] + 1.5*adm, self.body.position[1] + 0.3*adm
        )
        self.upper_arm.angle = np.pi*(53.13)/180
        
    