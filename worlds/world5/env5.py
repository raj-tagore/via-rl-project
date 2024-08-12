import pymunk
import numpy as np


class Environment():
    
    def __init__(self, rest_length=300, stiffness=100, damping=0, gravity=900):
        
        # constants
        self.REST_LENGTH_LIMITS = (100, 400)
        self.STIFFNESS_LIMITS = (20, 800)
        self.SURFACE_MASS = 1
        self.SURFACE_RADIUS = 15
        
        self.anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_shape = pymunk.Circle(self.anchor, 5)
        self.anchor.position = (150, 900)
        
        inertia = pymunk.moment_for_circle(self.SURFACE_MASS, 0, self.SURFACE_RADIUS)
        self.surface = pymunk.Body(self.SURFACE_MASS, inertia)
        self.surface_shape = pymunk.Circle(self.surface, 15)
        self.surface.position = (150, self.anchor.position[1] - rest_length)
        self.surface_previous_velocity = 0
        self.surface_acceleration = 0
        
        self.spring = pymunk.DampedSpring(self.anchor, self.surface, (0, 0), (0, 0), rest_length, stiffness, damping)
        self.gravity = gravity
        
    def reset(self, rest_length=300, stiffness=100, damping=0, gravity=900):
        
        self.anchor.position = (150, 900)
        
        self.surface.position = (150, self.anchor.position[1] - rest_length)
        self.surface_previous_velocity = 0
        self.surface_acceleration = 0
        
        self.spring.rest_length = rest_length
        self.spring.stiffness = stiffness
        self.spring.damping = damping
        
        self.gravity = gravity
        
    def update_accelerations(self):
        surface_velocity_change = self.surface.velocity.y - self.surface_previous_velocity
        self.surface_acceleration = surface_velocity_change
        self.surface_previous_velocity = self.surface.velocity.y