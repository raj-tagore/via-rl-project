import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np  

# Lets try having 2 springs on a groove joint
# It works! Everything is restricted to 1D

class Space:
    pygame.init()
    
    def __init__(self):
        
        self.robot_mass = 1
        self.robot_actuator_length = 100
        self.robot_actuator_stiffness = 100
        
        self.surface_length = 100
        self.surface_impedance = 300
        self.surface_damping = 0    
        self.gravity = 900
        
        self.screen_dim = (300, 1000)
        
        self.screen = pygame.display.set_mode(self.screen_dim)
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        self.space = pymunk.Space()
        self.space.gravity = (0, self.gravity)
        
        self.anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_shape = pymunk.Circle(self.anchor, 5)
        self.anchor.position = (self.screen_dim[0]//2, self.screen_dim[1] - 100)
        self.space.add(self.anchor, self.anchor_shape)
        
        self.surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        self.surface.position = (self.screen_dim[0]//2, 800)
        self.surface_shape = pymunk.Circle(self.surface, 5)
        self.space.add(self.surface, self.surface_shape)
        
        self.spring1 = pymunk.DampedSpring(self.anchor, self.surface, (0, 0), (0, 0), self.surface_length, self.surface_impedance, self.surface_damping)
        self.space.add(self.spring1)
        
        self.surface_groove = pymunk.GrooveJoint(self.anchor, self.surface, (0, 0), (0, -self.screen_dim[1]), (0, 0))
        self.space.add(self.surface_groove)
        
        self.robot_body = pymunk.Body(2, pymunk.moment_for_circle(1, 0, 25))
        self.robot_body.position = (150, 100)
        self.robot_shape = pymunk.Circle(self.robot_body, 25)
        self.space.add(self.robot_body, self.robot_shape)
        self.robot_body_groove = pymunk.GrooveJoint(self.anchor, self.robot_body, (0, 0), (0, -500), (0, 0))
        self.space.add(self.robot_body_groove)
        
        self.robot_EE = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        self.robot_EE.position = (150, 200)
        robot_shape = pymunk.Circle(self.robot_EE, 5)
        self.space.add(self.robot_EE, robot_shape)
        self.robot_EE_groove = pymunk.GrooveJoint(self.anchor, self.robot_EE, (0, 0), (0, -500), (0, 0))
        self.space.add(self.robot_EE_groove)
        
        self.robot_actuator = pymunk.DampedSpring(self.robot_body, self.robot_EE, (0, 0), (0, 0), 100, 100, 0)
        self.space.add(self.robot_actuator)
        
        self.steps = 0
        
    def step(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        self.space.step(1/100)
        self.steps += 1
        
    def _get_state(self):
        state = []
        # going from down to up, relevant information of all bodies is being added here
        state.append(self.anchor.position[1])
        state.append(self.surface.position[1])
        state.append(self.surface.velocity[1])
        state.append(self.robot_body.position[1])
        state.append(self.robot_body.velocity[1])
        state.append(self.robot_EE.position[1])  
        state.append(self.robot_EE.velocity[1])
        
        # now to append the robot actuator information, rest length and impedance
        state.append(self.robot_actuator.rest_length)
        state.append(self.robot_actuator.stiffness)
        
        return state      
    
    def _get_observation(self):
        return {
            'feet_contact': 1 if self.surface.position.y - self.robot_EE.position.y < 12 else 0,
            'self_actuator': np.array([self.robot_actuator.stiffness, self.robot_actuator.rest_length]),
            'self_body': np.array([self.robot_body.position.y, self.robot_body.velocity.y]),
        }
        
    def render(self):
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.update()
        self.clock.tick(60)
    
if __name__ == "__main__":
    space = Space()
    while True:
        space.step()
        space.render()
        print(space.clock)
    