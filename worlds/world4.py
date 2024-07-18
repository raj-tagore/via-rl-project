import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np  

# Extension of World 2 where I am having 2 springs on a groove joint
# Here I try to integrate a GUI with sliders to modify parameters of the environment in real time
# In this world, I am trying to use pygame_gui for the GUI

pygame.init()

class Robot:
    def __init__(self, mass_init=1, radius_init=25, length_init=100, length_limits=(25,200), 
                 impedance_init=100, impedance_limits=(0,1000), pos_init=(150, 100)):
        
        # The limits can be accessed anytime because they dont change
        self.length_limits = length_limits
        self.impedance_limits = impedance_limits
        
        # Robot's pymunk Body
        # position and velocity can be accessed by this body instance
        inertia_init = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.body = pymunk.Body(mass_init, inertia_init)
        self.body.position = pos_init 
        self.body_shape = pymunk.Circle(self.body, radius_init)
        
        # Robot's EE's Pymunk Body
        # EE's initial mass is 0.5 of the robot's mass
        self.EE = pymunk.Body(mass_init*0.5, pymunk.moment_for_circle(1, 0, 5))
        self.EE.position = (self.body.position[0], self.body.position[1] + length_init)
        self.EE_shape = pymunk.Circle(self.EE, 5)
        
        # Whether the EE is in contact with the environment
        self.EE_contact = False
        
        # Robot's Actuator Spring's Pymunk Spring
        # length and impedance can be accessed and modified by this spring instance
        self.actuator = pymunk.DampedSpring(self.body, self.EE, (0, 0), (0, 0), length_init, impedance_init, 0)
    
    def modify(self, **kwargs):
        # Modify the robot's properties
        for key, value in kwargs.items():
            if key == 'mass':
                self.body.mass = value
                self.EE.mass = value*0.5
            if key == 'radius':
                self.body_shape.radius = value
            if key == 'length':
                self.actuator.rest_length = value
            if key == 'impedance':
                self.actuator.stiffness = value
            if key == 'position':
                self.body.position = value
                
    def get_properties(self):
        return {'mass': self.body.mass, 
                'radius': self.body_shape.radius, 
                'rest_length': self.actuator.rest_length, 
                'impedance': self.actuator.stiffness, 
                'position': self.body.position, 
                'velocity': self.body.velocity}
            
class Environment:
    
    env_size = (1000, 300)
    
    def __init__(self, rest_length=100, impedance=100, damping=0, gravity=900):
        
        # These values dont change across a single episode
        self.anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_shape = pymunk.Circle(self.anchor, 5)
        self.anchor.position = (150, 900)
        
        # Surface has a mass of 1 and radius of 5
        self.surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        self.surface_shape = pymunk.Circle(self.surface, 5)
        self.surface.position = (150, self.anchor.position[1] - rest_length)
        
        # Spring connecting the anchor and the surface
        self.spring = pymunk.DampedSpring(self.anchor, self.surface, (0, 0), (0, 0), rest_length, impedance, damping)
        
        self.gravity = gravity
        
    def modify(self, **kwargs):
        # Modify the environment's properties
        for key, value in kwargs.items():
            if key == 'gravity':
                self.space.gravity = (0, value)
            if key == 'rest_length':
                self.spring.rest_length = value
            if key == 'impedance':
                self.spring.stiffness = value
            if key == 'damping':
                self.spring.damping = value
                
    def get_properties(self):
        return {'gravity': self.space.gravity, 
                'rest_length': self.spring.rest_length, 
                'impedance': self.spring.stiffness, 
                'damping': self.spring.damping,
                'position': self.surface.position,
                'velocity': self.surface.velocity}
        
class World:
    
    space = pymunk.Space()
    
    screen_dim = (1000, 1000)
    screen = pygame.display.set_mode(screen_dim)
    clock = pygame.time.Clock()
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    
    def __init__(self, robot, env):
        self.robot = robot
        self.env = env
        
        self.space.gravity = (0, self.env.gravity)
        
        # Add the anchor, surface and spring to the space
        self.space.add(env.anchor, env.anchor_shape, env.surface, env.surface_shape, env.spring)
        
        # Add the robot to the space
        self.space.add(robot.body, robot.body_shape, robot.EE, robot.EE_shape, robot.actuator)
        
        # Join everything to groove joints, and add the grooves to the space
        groove_end = (0, env.env_size[1]-env.anchor.position[1]*2)
        surface_groove = pymunk.GrooveJoint(env.anchor, env.surface, (0, 0), groove_end, (0, 0))
        robot_body_groove = pymunk.GrooveJoint(env.anchor, robot.body, (0, 0), groove_end, (0, 0))
        robot_EE_groove = pymunk.GrooveJoint(env.anchor, robot.EE, (0, 0), groove_end, (0, 0))
        self.space.add(surface_groove, robot_body_groove, robot_EE_groove)
        
    
        
        
    
            
            
            
            
            
            
    

