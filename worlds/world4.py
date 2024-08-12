import pymunk
import pymunk.pygame_util
import pygame
import sys
import numpy as np  
import gymnasium as gym
from gymnasium import spaces
import pygame_gui
from stable_baselines3 import PPO
import csv

# Extension of World 2 where I am having 2 springs on a groove joint
# Here I try to integrate a GUI with sliders to modify parameters of the environment in real time
# In this world, I am trying to use pygame_gui for the GUI
# I am also trying to implement a mathematical controller for the robot

class Robot:
    def __init__(self, mass=1, radius=25, length_init=100, length_limits=(50,300), 
                 impedance_init=100, impedance_limits=(50,400), pos_init=(150, 100)):
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
        self.body_previous_velocity = 0
        self.body_acceleration = 0
        # Robot's EE's Pymunk Body
        # EE's initial mass is 0.5 of the robot's mass
        self.EE = pymunk.Body(self.mass*0.5, pymunk.moment_for_circle(1, 0, 15))
        self.EE.position = (self.body.position[0], self.body.position[1] + length_init)
        self.EE_shape = pymunk.Circle(self.EE, 15)
        # Whether the EE is in contact with the environment
        self.EE_contact = 0
        self.EE_previous_velocity = 0
        self.EE_acceleration = 0
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
        self.body_previous_velocity = 0
        self.EE_previous_velocity = 0
        self.body_acceleration = 0
        self.EE_acceleration = 0
        
    def accelerations(self):
        body_velocity_change = self.body.velocity.y - self.body_previous_velocity
        self.body_acceleration = body_velocity_change
        EE_velocity_change = self.EE.velocity.y - self.EE_previous_velocity
        self.EE_acceleration = EE_velocity_change
        self.body_previous_velocity = self.body.velocity.y
        self.EE_previous_velocity = self.EE.velocity.y
        
        
             
class Environment():
    
    env_size = (1000, 300)
    
    def __init__(self, rest_length=300, impedance=100, damping=0, gravity=900):
        # These values dont change across a single episode
        self.anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_shape = pymunk.Circle(self.anchor, 5)
        self.anchor.position = (150, 900)
        # Surface has a mass of 1 and radius of 10
        self.surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 15))
        self.surface_shape = pymunk.Circle(self.surface, 15)
        self.surface.position = (150, self.anchor.position[1] - rest_length)
        self.surface_previous_velocity = 0
        self.surface_acceleration = 0
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
        self.surface_previous_velocity = 0
        self.surface_acceleration = 0
        
    def accelerations(self):
        surface_velocity_change = self.surface.velocity.y - self.surface_previous_velocity
        self.surface_acceleration = surface_velocity_change
        self.surface_previous_velocity = self.surface.velocity.y
        
                

class World(gym.Env):
    
    space = pymunk.Space()
    rate = 60
    
    def __init__(self, robot, env, mode='normal'):
        super(World, self).__init__()
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
        self.steps_per_episode = 100_000
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        # self.observation_space = spaces.Dict({
        #     'robot_position': spaces.Box(low=0, high=1000, shape=(1,)),
        #     'robot_velocity': spaces.Box(low=-1000, high=1000, shape=(1,)),
        #     'actuator_length': spaces.Box(low=self.robot.length_limits[0], high=self.robot.length_limits[1], shape=(1,)),
        #     'actuator_impedance': spaces.Box(low=self.robot.impedance_limits[0], high=self.robot.impedance_limits[1], shape=(1,)),
        #     'EE_contact': spaces.MultiBinary((1,))
        # })
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(9,))
        if mode == 'RL':
            pass
        elif mode == 'observe':
            pygame.init()
            self.screen_dim = (300, 1000)
            self.screen = pygame.display.set_mode(self.screen_dim)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
            self.file = open("current_run.csv", 'w', newline='')
            self.writer = csv.writer(self.file)
        elif mode == 'control':
            pygame.init()
            self.screen_dim = (1000, 1000)
            self.screen = pygame.display.set_mode(self.screen_dim)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
            self.manager = pygame_gui.UIManager(self.screen_dim)
            self.add_gui_elements()
            self.paused = False
            self.font = pygame.font.Font(None, 20)
            self.file = open("current_run.csv", 'w', newline='')
            self.writer = csv.writer(self.file)
        
    def step(self, action):
        self.robot.do_action(action)
        self.space.step(1/self.rate)
        self.robot.accelerations()
        self.env.accelerations()
        self.steps += 1
        observation = self.get_observation2()
        self.truncated = False if self.steps < self.steps_per_episode else True
        self.anti_cheat()
        reward = self.get_reward2()
        return observation, reward, self.terminated, self.truncated, {}
    
    def store_state(self):
        state = [self.steps,
                 self.robot.body.position.y,
                 self.robot.body.velocity.y,
                 self.robot.body_acceleration,
                 self.robot.EE.position.y,
                 self.robot.EE.velocity.y,
                 self.robot.EE_acceleration,
                 self.robot.EE_contact,
                 self.robot.actuator.rest_length,
                 self.robot.actuator.stiffness,
                 self.env.surface.position.y,
                 self.env.surface.velocity.y,
                 self.env.surface_acceleration]
        self.writer.writerow(state)
        
    def render(self):
        if self.mode == 'observe':
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
            self.screen.fill((255, 255, 255))
            self.space.debug_draw(self.draw_options)
            pygame.display.flip()
            self.clock.tick(self.rate)
            self.store_state()
        elif self.mode == "control":
            self.manager.update(self.clock.tick(self.rate)/1000.0)
            self.screen.fill((255, 255, 255))
            self.manager.draw_ui(self.screen)
            self.space.debug_draw(self.draw_options)
            
            text = f"Body Position: {self.robot.body.position.y}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 780))
            text = f"Body Velocity: {self.robot.body.velocity.y}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 800))
            text = f"Body Acceleration: {self.robot.body_acceleration}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 820))
            text = f"Actuator rest length: {self.robot.actuator.rest_length}"
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 840))
            text = f"Actuator Stiffness: {self.robot.actuator.stiffness}"   
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 860))
            text = f"Step: {self.steps}"   
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 880))
            text = "Rule Break: "+ str(self.truncated)  
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (400, 900))
            
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
        if self.terminated:
            reward = -10000
        return reward
    
    def get_reward2(self):
        # this reward function guides the robot slightly better
        # it rewards the robot more, if the robot maintains lesser velocity
        # it doesnt hand out rewards if the robot isnt in contact
        k = self.robot.EE_contact
        reward = k*np.exp(-((self.robot.body.velocity.y / 25) ** 2))
        if self.terminated:
            reward = -10000
        return reward
    
    def get_reward3(self):
        # this rewards robot for lesser accelerations
        reward = np.exp(-((self.robot.body_acceleration) ** 2))
        if self.terminated:
            reward = -10000
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
        
        observation = np.array([
            self.robot.body.position.y,
            self.robot.body.velocity.y,
            self.robot.actuator.rest_length,
            self.robot.actuator.stiffness,
            self.robot.EE_contact
        ], dtype=np.float32)
        return observation
    
    def get_observation1_dict(self):
        if self.env.surface.position.y - self.robot.EE.position.y < 5:
            self.robot.EE_contact = 1
        else:
            self.robot.EE_contact = 0
            
        observation = {
            'robot_position': np.array([self.robot.body.position.y],dtype=np.float32),
            'robot_velocity': np.array([self.robot.body.velocity.y],dtype=np.float32),
            'actuator_length': np.array([self.robot.actuator.rest_length],dtype=np.float32),
            'actuator_impedance': np.array([self.robot.actuator.stiffness],dtype=np.float32),
            'EE_contact': np.array([self.robot.EE_contact],dtype=np.int8)
        }
        return observation
    
    def get_observation2(self):
        # In this Observation Function Robot observes:
        # accelerations of body
        # position, velocity, acceleration of EE
        # alongside all the observations from get_observation1
        
        if self.env.surface.position.y - self.robot.EE.position.y < 5:
            self.robot.EE_contact = 1
        else:
            self.robot.EE_contact = 0
            
        observation = np.array([
            self.robot.body.position.y,
            self.robot.body.velocity.y,
            self.robot.body_acceleration,
            self.robot.actuator.rest_length,
            self.robot.actuator.stiffness,
            self.robot.EE.position.y,
            self.robot.EE.velocity.y,
            self.robot.EE_acceleration,
            self.robot.EE_contact
        ], dtype=np.float32)
        
        return observation
            
    def anti_cheat(self):
        # This function checks if the robot breaks the rules
        # and if it does, it punishes the robot
        if self.robot.body.position.y > self.robot.EE.position.y:
            self.terminated = True
        if self.robot.EE.position.y > self.env.surface.position.y:
            self.terminated = True
    
    def reset(self, seed=42):
        np.random.seed(seed)
        
        # Limits to randomize between
        # robot_pos_y_limits = (25, 500)
        # env_rest_length_limits = (100, 300)
        # env_stiffness_limits = (20, 800)
        # env_damping_limits = (0, 10)
        # env_gravity_limits = (100, 2000)
        # robot_start_pos = (150, np.random.uniform(robot_pos_y_limits[0], robot_pos_y_limits[1]))
        # env_rest_length = np.random.uniform(env_rest_length_limits[0], env_rest_length_limits[1])
        # env_stiffness = np.random.uniform(env_stiffness_limits[0], env_stiffness_limits[1])
        # env_damping = np.random.uniform(env_damping_limits[0], env_damping_limits[1])
        # env_gravity = np.random.uniform(env_gravity_limits[0], env_gravity_limits[1])
        # self.robot.reset(pos_init=robot_start_pos)
        # self.env.reset(rest_length=env_rest_length, impedance=env_stiffness, damping=0, gravity=env_gravity)
        
        self.robot.reset()
        self.env.reset()
        
        self.steps = 0
        self.terminated = False
        self.truncated = False
        return self.get_observation2(), {}
    
    def close(self):
        pygame.quit()
        sys.exit()        
        
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
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((600, 700), (100, 50)),
            text='Reset', manager=self.manager)
    
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
                elif event.ui_element == self.reset_button:
                    seed = np.random.randint(0, 1000)
                    self.reset(seed=seed)
            self.manager.process_events(event)
        if self.paused == False:
            self.space.step(1/self.rate)
            self.robot.accelerations()
            self.env.accelerations()
            self.steps += 1
            self.anti_cheat()   
            self.store_state()
     
    def mathematical_controller(self):
        m = self.robot.mass
        g = self.env.gravity
        x = self.robot.actuator.rest_length-(self.robot.EE.position.y-self.robot.body.position.y)
        v = self.robot.body.velocity.y
        k = self.robot.actuator.stiffness
        alpha = 0.03
        beta = 0.2
        v_ee = self.robot.EE.velocity.y
        a_ee = self.robot.EE_acceleration
        if v_ee*a_ee>0 and abs(x)>10:
            new_k = m*g/x
            if new_k<self.robot.impedance_limits[0]:
                new_k = self.robot.impedance_limits[0]
            elif new_k>self.robot.impedance_limits[1]:
                new_k = self.robot.impedance_limits[1]
            self.robot.actuator.stiffness = new_k
        elif v_ee*a_ee<0 and abs(k)>10:
            new_x = m*g/k
            new_rest_length = new_x + self.robot.EE.position.y - self.robot.body.position.y
            if new_rest_length<self.robot.length_limits[0]:
                new_rest_length = self.robot.length_limits[0]
            elif new_rest_length>self.robot.length_limits[1]:
                new_rest_length = self.robot.length_limits[1]
            self.robot.actuator.rest_length = new_rest_length
            

        
if __name__ == "__main__":
    robot = Robot()
    env = Environment()
    env.reset()
    
    MODE = 'observe'
    
    if MODE == 'observe':
        world = World(robot, env, mode='observe')
        model = PPO.load("models/model14")
        while True:
            action, _ = model.predict(world.get_observation2())
            world.step(action)
            if world.truncated or world.terminated:
                world.reset() 
                print("Reset")
            world.render()
                
    elif MODE == 'control':        
        world = World(robot, env, mode='control')
        while True:
            world.control()
            world.mathematical_controller()
            world.render()
    
        
    
        
        
    
            
            
            
            
            
            
    

