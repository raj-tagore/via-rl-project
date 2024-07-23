import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np  
import pymunk.matplotlib_util
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# Extension of World 2 where I am having 2 springs on a groove joint
# Here I try to integrate a GUI with sliders to modify parameters of the environment in real time
# In this world, I am trying to use pygame_gui for the GUI

pygame.init()

class Robot:
    def __init__(self, mass=1, radius=25, length_init=100, length_limits=(25,200), 
                 impedance_init=100, impedance_limits=(0,1000), pos_init=(150, 100)):
        # These can be accessed anytime because they dont change
        self.mass = mass
        self.radius = radius
        self.length_limits = length_limits
        self.impedance_limits = impedance_limits
        # Robot's pymunk Body
        # position and velocity can be accessed by this body instance
        inertia_init = pymunk.moment_for_circle(self.mass, 0, self.radius)
        self.body = pymunk.Body(self.mass, inertia_init)
        self.body.position = pos_init 
        self.body_shape = pymunk.Circle(self.body, self.radius)
        # Robot's EE's Pymunk Body
        # EE's initial mass is 0.5 of the robot's mass
        self.EE = pymunk.Body(self.mass*0.5, pymunk.moment_for_circle(1, 0, 15))
        self.EE.position = (self.body.position[0], self.body.position[1] + length_init)
        self.EE_shape = pymunk.Circle(self.EE, 15)
        # Whether the EE is in contact with the environment
        self.EE_contact = 0
        # Robot's Actuator Spring's Pymunk Spring
        # length and impedance can be accessed and modified by this spring instance
        self.actuator = pymunk.DampedSpring(self.body, self.EE, (0, 0), (0, 0), length_init, impedance_init, 0)
                 
    def do_action(self, action):
        action = np.array(action)
        action[0] = action[0]*(self.length_limits[1] - self.length_limits[0])/2 + (self.length_limits[1] + self.length_limits[0])/2
        action[1] = action[1]*(self.impedance_limits[1] - self.impedance_limits[0])/2 + (self.impedance_limits[1] + self.impedance_limits[0])/2
        self.actuator.rest_length = action[0]
        self.actuator.stiffness = action[1]
        
    def reset(self, pos_init=(150, 100), length_init=100, impedance_init=100):
        self.body.position = pos_init
        self.EE.position = (self.body.position[0], self.body.position[1] + length_init)
        self.actuator.rest_length = length_init
        self.actuator.stiffness = impedance_init
        
        
             
class Environment:
    
    env_size = (1000, 300)
    
    def __init__(self, rest_length=100, impedance=100, damping=0, gravity=900):
        # These values dont change across a single episode
        self.anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_shape = pymunk.Circle(self.anchor, 5)
        self.anchor.position = (150, 900)
        # Surface has a mass of 1 and radius of 10
        self.surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 15))
        self.surface_shape = pymunk.Circle(self.surface, 15)
        self.surface.position = (150, self.anchor.position[1] - rest_length)
        # Spring connecting the anchor and the surface
        self.spring = pymunk.DampedSpring(self.anchor, self.surface, (0, 0), (0, 0), rest_length, impedance, damping)
        self.gravity = gravity
        
    def reset(self, rest_length=100, impedance=100, damping=0, gravity=900):
        self.anchor.position = (150, 900)
        self.surface.position = (150, self.anchor.position[1] - rest_length)
        self.spring.rest_length = rest_length
        self.spring.stiffness = impedance
        self.spring.damping = damping
        self.gravity = gravity
                
        
        
class World:
    
    space = pymunk.Space()
    rate = 60
    
    def __init__(self, robot, env, mode='normal'):
        self.robot = robot
        self.env = env
        self.space.gravity = (0, self.env.gravity)
        self.mode = mode
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
        # More information about the world
        self.steps = 0
        self.stable_steps = 0
        self.terminated = False
        self.truncated = False
        self.steps_per_episode = 50000
        if mode == 'RL':
            pass
        elif mode == 'observe':
            self.screen_dim = (300, 1000)
            self.screen = pygame.display.set_mode(self.screen_dim)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
            self.fig, self.axes = plt.subplots(4,1)
            self.init_plots()
            # thread = threading.Thread(target=self.show_plots)
            # thread.start()  
            # thread.join()
        elif mode == 'control':
            self.screen_dim = (1000, 1000)
            self.screen = pygame.display.set_mode(self.screen_dim)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
            self.manager = pygame_gui.UIManager(self.screen_dim)
            self.add_gui_elements()
            self.paused = False
        
    def step(self, action):
        self.robot.do_action(action)
        self.space.step(1/self.rate)
        self.steps += 1
        observation = self.get_observation1()
        reward = self.get_reward1()
        # The episode terminates if the robot EE goes below surface
        if self.env.surface.position.y - self.robot.EE.position.y < 0:
            self.terminate = True
        self.truncated = False if self.steps < self.steps_per_episode else True
        return observation, reward, self.terminated, self.truncated, {}
        
    def render(self):
        if self.mode == 'observe':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            self.screen.fill((0, 0, 0))
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(self.rate)
        elif self.mode == "control":
            self.manager.update(self.clock.tick(self.rate)/1000.0)
            self.screen.fill((0, 0, 0))
            self.manager.draw_ui(self.screen)
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(self.rate)
        
    def get_reward1(self):
        # this reward function rewards 1 point if the robot is stable for 200 steps
        # Total reward the robot can gain is 50000/200 = 250
        if self.robot.body.velocity.y < 10:
            self.stable_steps += 1
        else:
            self.stable_steps = 0  
        if self.stable_steps > 200:
            reward = 1
            self.stable_steps = 0
        else:
            reward = 0
        return reward
    
    def get_observation1(self):
        # In this Observation Function Robot observes: 
        # position, velocity of the body
        # length, impedance of the actuator
        # Whether the EE is in contact with the environment
        if self.env.surface.position.y - self.robot.EE.position.y < 5:
            self.robot.EE_contact = 1
        else:
            self.robot.EE_contact = 0
            
        observation = np.array([self.robot.body.position.y, 
                                self.robot.body.velocity.y, 
                                self.robot.actuator.rest_length, 
                                self.robot.actuator.stiffness, 
                                self.robot.EE_contact])
        return observation
    
    def reset(self, seed=42):
        np.random.seed(seed)
        self.robot.reset()
        self.env.reset()
        self.steps = 0
        self.terminated = False
        self.truncated = False
        return self.get_observation1(), {}
    
    def close(self):
        pygame.quit()
        sys.exit()
    
    # Code for plotting and stuff here
    def init_plots(self):
        self.xdata = []
        self.robot_pos_ydata = []
        self.robot_vel_ydata = []
        self.robot_length_ydata = []
        self.robot_impedance_ydata = []
        
    def update_plots(self, _):
        self.xdata.append(self.steps)
        self.robot_pos_ydata.append(self.robot.body.position[1])
        self.robot_vel_ydata.append(self.robot.body.velocity[1])
        self.robot_length_ydata.append(self.robot.actuator.rest_length)
        self.robot_impedance_ydata.append(self.robot.actuator.stiffness)
        if len(self.xdata) > 100:
            self.xdata.pop(0)
            self.robot_pos_ydata.pop(0)
            self.robot_vel_ydata.pop(0)
            self.robot_length_ydata.pop(0)
            self.robot_impedance_ydata.pop(0)
        for ax in self.axes:
            ax.set_xlim(max(0, self.steps-100), self.steps)
        self.axes[0].plot(self.xdata, self.robot_pos_ydata)
        self.axes[1].plot(self.xdata, self.robot_vel_ydata)
        self.axes[2].plot(self.xdata, self.robot_length_ydata)
        self.axes[3].plot(self.xdata, self.robot_impedance_ydata)
        
    def show_plots(self):
        anim = FuncAnimation(self.fig, self.update_plots, frames=self.steps, blit=False)
        plt.tight_layout()
        plt.show()
        
    # Code for UI and stuff here
    def add_gui_elements(self):
        self.robot_length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 100), (400, 25)), 
            start_value=np.mean(robot.length_limits), 
            value_range=robot.length_limits, manager=self.manager)
        self.robot_length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 150), (400, 25)), 
            text=f'Robot Length: {robot.actuator.rest_length}', manager=self.manager)
        self.robot_impedance_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 200), (400, 25)),
            start_value=np.mean(robot.impedance_limits), 
            value_range=robot.impedance_limits, manager=self.manager)
        self.robot_impedance_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 250), (400, 25)), 
            text=f'Robot Impedance: {robot.actuator.stiffness}', manager=self.manager)
        self.surface_length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 300), (400, 25)),
            start_value=env.spring.rest_length, 
            value_range=(0, 800), manager=self.manager)
        self.surface_length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 350), (400, 25)),
            text=f'Surface Length: {env.spring.rest_length}', manager=self.manager)
        self.surface_impedance_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 400), (400, 25)),
            start_value=env.spring.stiffness, 
            value_range=(0, 1000), manager=self.manager)
        self.surface_impedance_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 450), (400, 25)),
            text=f'Surface Impedance: {env.spring.stiffness}', manager=self.manager)
        self.robot_position_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 500), (400, 25)),
            start_value=robot.body.position[1], 
            value_range=(0, 800), manager=self.manager)
        self.robot_position_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 550), (400, 25)),
            text=f'Robot Position: {robot.body.position[1]}', manager=self.manager)
        self.surface_position_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 600), (400, 25)),
            start_value=env.surface.position[1], 
            value_range=(0, 800), manager=self.manager)
        self.surface_position_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 650), (400, 25)),
            text=f'Surface Position: {env.surface.position[1]}', manager=self.manager)
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 700), (100, 50)),
            text='Pause', manager=self.manager)
    
    def control(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == self.robot_length_slider:
                    self.robot.actuator.rest_length = event.value
                    self.robot.EE.position = (self.robot.EE.position[0], self.robot.body.position[1] + event.value)
                    self.robot_length_label.set_text(f'Robot Length: {event.value}')
                elif event.ui_element == self.robot_impedance_slider:
                    self.robot.actuator.stiffness = event.value
                    self.robot_impedance_label.set_text(f'Robot Impedance: {event.value}')
                elif event.ui_element == self.surface_length_slider:
                    self.env.spring.rest_length = event.value
                    self.env.surface.position = (self.env.surface.position[0], self.env.anchor.position[1] - event.value)
                    self.surface_length_label.set_text(f'Surface Length: {event.value}')
                elif event.ui_element == self.surface_impedance_slider:
                    self.env.spring.stiffness = event.value
                    self.surface_impedance_label.set_text(f'Surface Impedance: {event.value}')
                elif event.ui_element == self.robot_position_slider:
                    self.robot.body.position = (self.robot.body.position[0], event.value)
                    self.robot.EE.position = (self.robot.EE.position[0], event.value + self.robot.actuator.rest_length)
                    self.robot_position_label.set_text(f'Robot Position: {event.value}')
                elif event.ui_element == self.surface_position_slider:
                    self.env.surface.position = (self.env.surface.position[0], event.value)
                    self.surface_position_label.set_text(f'Surface Position: {event.value}')
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.pause_button:
                    self.paused = not self.paused
            self.manager.process_events(event)
        if self.paused == False:
            self.space.step(1/self.rate)
            self.steps += 1
        observation = self.get_observation1()
        reward = self.get_reward1()
        # The episode terminates if the robot EE goes below surface
        if self.env.surface.position.y - self.robot.EE.position.y < 0:
            self.terminate = True
        self.truncated = False if self.steps < self.steps_per_episode else True
        return observation, reward, self.terminated, self.truncated, {}
        

        
        
if __name__ == "__main__":
    robot = Robot()
    env = Environment()
    world = World(robot, env, mode='observe')
    while True:
        world.step([0, 0])
        # world.control()
        world.render()
    
        
    
        
        
    
            
            
            
            
            
            
    

