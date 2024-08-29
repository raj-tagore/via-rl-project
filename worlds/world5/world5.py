import sys
import os
sys.path.append('worlds/world5')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pymunk
import pymunk.pygame_util
import pygame
import numpy as np  
import gymnasium as gym
from gymnasium import spaces
import pygame_gui
from stable_baselines3 import PPO, DDPG, A2C, SAC
import csv
from robot5 import Robot
from env5 import Environment
import matplotlib.pyplot as plt

# Cleaner version of world4. 
# Here I implement stuff like adding noise to observations and implementing a more realistic
# control methodology than the current 100% accurate and perfect control.
# The code will also be more modular in different files, so more understandable and
# hence, more editable.


class World(gym.Env):
    
    space = pymunk.Space()
    FPS = 60
    SIZE = (1000,1000)
    
    def __init__(self, robot, env, mode='RL'):
        super(World, self).__init__()
        
        self.robot = robot
        self.env = env
        self.mode = mode
        self.space.gravity = (0, self.env.gravity)
        
        self.space.add(self.env.anchor, self.env.anchor_shape, 
                       self.env.surface, self.env.surface_shape, self.env.spring)  
        self.space.add(self.robot.body, self.robot.body_shape, 
                       self.robot.EE, self.robot.EE_shape, self.robot.actuator)
        
        groove_end = (0, self.SIZE[1]-env.anchor.position[1]*2)
        surface_groove = pymunk.GrooveJoint(env.anchor, env.surface, (0, 0), groove_end, (0, 0))
        robot_body_groove = pymunk.GrooveJoint(env.anchor, robot.body, (0, 0), groove_end, (0, 0))
        robot_EE_groove = pymunk.GrooveJoint(env.anchor, robot.EE, (0, 0), groove_end, (0, 0))
        self.space.add(surface_groove, robot_body_groove, robot_EE_groove)
        
        self.steps = 0
        self.stable_steps = 0
        self.terminated = False
        self.truncated = False
        self.steps_per_episode = 2048
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        # obs_lows = np.array([0, -5000, -5000, 0, 0, 0, -5000, -5000, 0], dtype=np.float32) 
        # obs_highs = np.array([1000, 5000, 5000, 500, 1000, 1000, 5000, 5000, 1], dtype=np.float32)
        # self.observation_space = spaces.Box(low=obs_lows, high=obs_highs, shape=(9,))
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        self.total_reward = 0
        self.avg_reward = 0
        
        self.K_CONTROL = 'incremental'
        self.L_CONTROL = 'incremental'
        
        if mode == 'RL':
            pass
        else:
            pygame.init()
            self.screen = pygame.display.set_mode(self.SIZE)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.clock = pygame.time.Clock()
            
            self.file = open('current_run.csv', 'w', newline='')    
            self.writer = csv.writer(self.file)
            
            self.manager = pygame_gui.UIManager(self.SIZE)
            self.font = pygame.font.Font(None, 20)
            self.paused = False
            self.add_gui_elements()
    
    def step(self, action):
        self.robot.do_action(action, noise=False, l_control=self.L_CONTROL, k_control=self.K_CONTROL)
        self.space.step(1/self.FPS)
        self.steps += 1
        observation = self.get_observation()
        self.truncated = self.steps >= self.steps_per_episode
        self.anti_cheat()
        reward = self.reward = self.get_reward()
        return observation, reward, self.terminated, self.truncated, {}
    
    def update(self, action=None):
        if self.mode == 'RL':
            if action is not None:
                self.robot.do_action(action, noise=False, l_control=self.L_CONTROL, k_control=self.K_CONTROL)
            self.space.step(1/self.FPS)
            self.steps += 1
            observation = self.get_observation()
            self.truncated = self.steps >= self.steps_per_episode
            self.anti_cheat()
            reward = self.reward = self.get_reward()
            return observation, reward, self.terminated, self.truncated, {}
        else:
            if action is not None:
                self.robot.do_action(action, noise=False, l_control=self.L_CONTROL, k_control=self.K_CONTROL)
            self.gui_control()
            if not self.paused:
                self.space.step(1/self.FPS)
                self.steps += 1
                observation = self.get_observation()
                self.truncated = self.steps >= self.steps_per_episode
                self.anti_cheat()
                reward = self.reward = self.get_reward()
                return observation, reward, self.terminated, self.truncated, {}
            
    def get_observation(self):
        OBSERVATION_FUNCTION = 4
        self.robot.update_accelerations()
        self.env.update_accelerations()
        if self.env.surface.position.y - self.robot.EE.position.y < 30:
            self.robot.EE_contact = 1
        else:
            self.robot.EE_contact = 0
            
        match OBSERVATION_FUNCTION:
            case 1:
                observation = np.array([
                    self.robot.body.position.y,
                    self.robot.body.velocity.y,
                    self.robot.actuator.rest_length,
                    self.robot.actuator.stiffness,
                    self.robot.EE_contact
                ], dtype=np.float32)
            case 2:
                observation = np.array([
                    self.robot.body.position.y,
                    self.robot.body.velocity.y,
                    self.robot.body_acceleration,
                    self.robot.EE.position.y-self.robot.body.position.y,
                    self.robot.actuator.stiffness,
                    self.robot.EE.position.y,
                    self.robot.EE.velocity.y,
                    self.robot.EE_acceleration,
                    self.robot.EE_contact
                ], dtype=np.float32)
            case 3:
                # Add noise to the observations
                observation = np.array([
                    self.robot.body.position.y + np.random.normal(0, 0.1*self.obs_highs[0]),
                    self.robot.body.velocity.y + np.random.normal(0, 0.1*self.obs_highs[1]),
                    self.robot.body_acceleration + np.random.normal(0, 0.1*self.obs_highs[2]),
                    self.robot.actuator.rest_length + np.random.normal(0, 0.1*self.obs_highs[3]),
                    self.robot.actuator.stiffness + np.random.normal(0, 0.1*self.obs_highs[4]),
                    self.robot.EE.position.y + np.random.normal(0, 0.1*self.obs_highs[5]),
                    self.robot.EE.velocity.y + np.random.normal(0, 0.1*self.obs_highs[6]),
                    self.robot.EE_acceleration + np.random.normal(0, 0.1*self.obs_highs[7]),
                    self.robot.EE_contact
                ], dtype=np.float32)
            case 4:
                observation = np.array([
                    self.robot.body_acceleration,
                    self.robot.EE.position.y-self.robot.body.position.y,
                    self.robot.EE.velocity.y-self.robot.body.velocity.y,
                    self.robot.actuator.rest_length,
                    self.robot.actuator.stiffness,
                    self.robot.EE_acceleration,
                    self.robot.EE_contact,
                ], dtype=np.float32)
        return observation
    
    def anti_cheat(self):
        self.out_of_bounds = False
        if self.robot.body.position.y > self.robot.EE.position.y:
            self.out_of_bounds = True
            self.terminated = True
        if self.robot.EE.position.y > self.env.surface.position.y:
            self.out_of_bounds = True
            self.terminated = True
        if self.env.surface.position.y > self.env.anchor.position.y:
            self.out_of_bounds = True
            self.terminated = True
            
    def get_reward(self):
        REWARD_FUNCTION = 1
        match REWARD_FUNCTION:
            case 1:
                if self.robot.body.velocity.y < 10 and self.robot.EE_contact:
                    self.stable_steps += 1
                else:
                    self.stable_steps = 0  
                if self.stable_steps > 20:
                    reward = 1
                else:
                    reward = 0
            case 2:
                reward = np.exp(-((self.robot.body.velocity.y / 25) ** 2))
            case 3:
                reward = np.exp(-((self.robot.body_acceleration) ** 2))
            case 4:
                reward = 1 if self.robot.body.velocity.y < 10 else 0
            case 5:
                reward = np.exp(-((self.robot.body.velocity.y / 25) ** 2)) + np.exp(-((self.robot.EE.velocity.y / 25) ** 2))
            case 6:
                v = self.robot.body.velocity.y
                reward = (np.exp(-(v/70)**2+5) - 60)/10
            case 7:
                reward = 1
        
        if self.out_of_bounds:
            reward = -100
            
        self.total_reward += reward
        self.avg_reward = self.total_reward/self.steps
        return reward
    
    def reset(self, seed=42):
        RANDOM_ON_RESET = True
        np.random.seed(seed)
        
        prev_ep_reward = self.total_reward
        
        robot_mass_limits = (1, 5)
        robot_pos_y_limits = (100, 300)
        
        surface_mass_limits = (1, 2)
        env_rest_length_limits = (100, 300)
        env_stiffness_multiplier_limits = (50, 150)
        
        if RANDOM_ON_RESET:
            robot_mass = np.random.uniform(robot_mass_limits[0], robot_mass_limits[1])
            robot_start_pos = (150, np.random.uniform(robot_pos_y_limits[0], robot_pos_y_limits[1]))
            
            surface_mass = np.random.uniform(surface_mass_limits[0], surface_mass_limits[1])
            env_rest_length = np.random.uniform(env_rest_length_limits[0], env_rest_length_limits[1])
            env_stiffness_multiplier = np.random.uniform(env_stiffness_multiplier_limits[0], env_stiffness_multiplier_limits[1])
            
            self.robot.reset(mass=robot_mass, pos_init=robot_start_pos)
            self.env.reset(mass=surface_mass, rest_length=env_rest_length, stiffness_multiplier=env_stiffness_multiplier)
        else:
            self.robot.reset()
            self.env.reset()
        
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.total_reward = 0
        return self.get_observation(), {'reward': prev_ep_reward}
    
    def close(self):
        self.file.close()
        pygame.quit()
        sys.exit()
        
    def render(self):
        self.manager.update(1/self.FPS)
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        self.manager.draw_ui(self.screen)
        
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
        text = "Out of bounds: "+ str(self.out_of_bounds)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (400, 900))
        text = "Reward: "+ str(self.reward)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (400, 920))
        text = "Robot mass: "+ str(self.robot.body.mass)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 780))
        text = "Env Stiffness: "+ str(self.env.spring.stiffness)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 800))
        text = "Avg Reward per Step: "+ str(self.avg_reward)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 820))
        
        self.clock.tick(self.FPS)
        pygame.display.flip()
        self.store_state()
        
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
        
    def add_gui_elements(self):
        self.robot_length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 100), (400, 25)), 
            start_value=np.mean(self.robot.LENGTH_LIMITS), 
            value_range=self.robot.LENGTH_LIMITS, 
            manager=self.manager
            )
        self.robot_length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 150), (400, 25)), 
            text=f'Robot Length: {self.robot.actuator.rest_length}', 
            manager=self.manager
            )
        self.robot_impedance_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 200), (400, 25)),
            start_value=np.mean(self.robot.STIFFNESS_LIMITS), 
            value_range=self.robot.STIFFNESS_LIMITS, 
            manager=self.manager
            )
        self.robot_impedance_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 250), (400, 25)), 
            text=f'Robot Impedance: {self.robot.actuator.stiffness}', 
            manager=self.manager
            )
        self.surface_length_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 300), (400, 25)),
            start_value=np.mean(self.env.REST_LENGTH_LIMITS), 
            value_range=self.env.REST_LENGTH_LIMITS, 
            manager=self.manager
            )
        self.surface_length_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 350), (400, 25)),
            text=f'Surface Length: {self.env.spring.rest_length}', 
            manager=self.manager
            )
        self.surface_impedance_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 400), (400, 25)),
            start_value=np.mean(self.env.STIFFNESS_LIMITS), 
            value_range=(0, 1000), 
            manager=self.manager
            )
        self.surface_impedance_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 450), (400, 25)),
            text=f'Surface Impedance: {self.env.spring.stiffness}', 
            manager=self.manager
            )
        self.robot_position_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 500), (400, 25)),
            start_value=self.robot.body.position[1], 
            value_range=(0, 800), 
            manager=self.manager
            )
        self.robot_position_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 550), (400, 25)),
            text=f'Robot Position: {self.robot.body.position.y}', 
            manager=self.manager
            )
        self.surface_position_slider = pygame_gui.elements.UIHorizontalSlider(
            relative_rect=pygame.Rect((400, 600), (400, 25)),
            start_value=self.env.surface.position[1], 
            value_range=(0, 800), 
            manager=self.manager
            )
        self.surface_position_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((400, 650), (400, 25)),
            text=f'Surface Position: {self.env.surface.position.y}', 
            manager=self.manager
            )
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((400, 700), (100, 50)),
            text='Pause', 
            manager=self.manager
            )
        self.reset_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((600, 700), (100, 50)),
            text='Reset', 
            manager=self.manager
            )
    
    def gui_control(self):
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
    
    def mathematical_control(self):
        RETURN_ACTION = True
        L_CONTROL = self.L_CONTROL
        K_CONTROL = self.K_CONTROL
        
        m = self.robot.body.mass
        g = self.env.gravity
        x = self.robot.actuator.rest_length-(self.robot.EE.position.y-self.robot.body.position.y)
        v = self.robot.body.velocity.y
        k = self.robot.actuator.stiffness
        alpha = 30
        beta = 0.2
        v_ee = self.robot.EE.velocity.y
        a_ee = self.robot.EE_acceleration
        
        l_max = self.robot.LENGTH_LIMITS[1]
        l_min = self.robot.LENGTH_LIMITS[0]
        k_max = self.robot.STIFFNESS_LIMITS[1]
        k_min = self.robot.STIFFNESS_LIMITS[0]
        
        if not RETURN_ACTION:
            if v_ee*a_ee>0 and abs(x)>10:
                new_k = m*g/x
                new_k = np.clip(new_k, k_min, k_max)
                self.robot.actuator.stiffness = new_k
            elif v_ee*a_ee<0 and abs(k)>10:
                new_x = m*g/k
                new_rest_length = new_x + self.robot.EE.position.y - self.robot.body.position.y
                new_rest_length = np.clip(new_rest_length, l_min, l_max)
                self.robot.actuator.rest_length = new_rest_length    
        else:
            # Modify K
            if v_ee*a_ee>0 and abs(x)>10:
                new_k = m*g/x
                new_k = np.clip(new_k, k_min, k_max)
                if K_CONTROL == 'direct':
                    normalized_k = (new_k - ((k_max+k_min)/2))/((k_max-k_min)/2)
                elif K_CONTROL == 'incremental':
                    normalized_k = alpha*(new_k - k)/(k_max-k_min)
                if L_CONTROL == 'direct':
                    normalized_x = (self.robot.actuator.rest_length - ((l_max+l_min)/2))/((l_max-l_min)/2)
                elif L_CONTROL == 'incremental':
                    normalized_x = 0
                normalized_x = np.clip(normalized_x, -1, 1)
                normalized_k = np.clip(normalized_k, -1, 1)
                action = np.array([normalized_x, normalized_k])
                return action
            # Modify X
            elif v_ee*a_ee<0 and abs(k)>10:
                new_x = m*g/k
                new_rest_length = new_x + self.robot.EE.position.y - self.robot.body.position.y
                new_rest_length = np.clip(new_rest_length, l_min, l_max)
                if K_CONTROL == 'direct':
                    normalized_k = (k - ((k_max+k_min)/2))/((k_max-k_min)/2)
                elif K_CONTROL == 'incremental':
                    normalized_k = 0
                if L_CONTROL == 'direct':
                    normalized_x = (new_rest_length - ((l_max+l_min)/2))/((l_max-l_min)/2)
                elif L_CONTROL == 'incremental':
                    normalized_x = alpha*(new_rest_length - self.robot.actuator.rest_length)/(l_max-l_min)
                normalized_x = np.clip(normalized_x, -1, 1)
                normalized_k = np.clip(normalized_k, -1, 1)
                action = np.array([normalized_x, normalized_k])
                return action
        return None
            
    def only_k_control(self):
        m = self.robot.body.mass
        g = self.env.gravity
        x = self.robot.actuator.rest_length-(self.robot.EE.position.y-self.robot.body.position.y)
        k = self.robot.actuator.stiffness
        v_ee = self.robot.EE.velocity.y
        a_ee = self.robot.EE_acceleration
        alpha = 30
        
        l_max = self.robot.LENGTH_LIMITS[1]
        l_min = self.robot.LENGTH_LIMITS[0]
        k_max = self.robot.STIFFNESS_LIMITS[1]
        k_min = self.robot.STIFFNESS_LIMITS[0]
        
        if abs(x)>10:
            new_k = m*g/x
            new_k = np.clip(new_k, k_min, k_max)
            normalized_k = alpha*(new_k - k)/(k_max-k_min)
            normalized_k = np.clip(normalized_k, -1, 1)
            return np.array([0, normalized_k])
        return None
            
    def only_x_control(self):
        m = self.robot.body.mass
        g = self.env.gravity
        x = self.robot.actuator.rest_length-(self.robot.EE.position.y-self.robot.body.position.y)
        k = self.robot.actuator.stiffness
        v_ee = self.robot.EE.velocity.y
        a_ee = self.robot.EE_acceleration
        alpha = 30
        
        l_max = self.robot.LENGTH_LIMITS[1]
        l_min = self.robot.LENGTH_LIMITS[0]
        k_max = self.robot.STIFFNESS_LIMITS[1]
        k_min = self.robot.STIFFNESS_LIMITS[0]
        
        if abs(x)>10:
            new_x = m*g/k
            new_rest_length = new_x + self.robot.EE.position.y - self.robot.body.position.y
            new_rest_length = np.clip(new_rest_length, l_min, l_max)
            normalized_x = alpha*(new_rest_length - self.robot.actuator.rest_length)/(l_max-l_min)
            normalized_x = np.clip(normalized_x, -1, 1)
            return np.array([normalized_x, 0])
        return None
            
    def random_control(self):
        action = np.random.uniform(-1, 1, size=(2,))
        return action
    
    def mathematical_control2(self):
        m = self.robot.body.mass
        g = self.env.gravity
        x = self.robot.actuator.rest_length-(self.robot.EE.position.y-self.robot.body.position.y)
        v = self.robot.body.velocity.y
        k = self.robot.actuator.stiffness
        v_ee = self.robot.EE.velocity.y
        a_ee = self.robot.EE_acceleration
        
        l_max = self.robot.LENGTH_LIMITS[1]
        l_min = self.robot.LENGTH_LIMITS[0]
        k_max = self.robot.STIFFNESS_LIMITS[1]
        k_min = self.robot.STIFFNESS_LIMITS[0]
        
                
    
    
if __name__ == '__main__':
    robot = Robot()
    env = Environment()
    world = World(robot, env, mode='control')
    
    MODE = 'RL'
    
    if MODE == 'RL':
        # model = PPO.load("models/PPO/model54")
        # model = A2C.load("models/A2C/model2")
        model = SAC.load("models/SAC/model4")
        while True:
            action, _ = model.predict(world.get_observation())
            world.update(action=action)
            world.render()
    else:
        while True:
            action = world.mathematical_control()
            world.update(action=action)
            world.render()
                
                
        
        


