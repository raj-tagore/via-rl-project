import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pymunk
import pymunk.pygame_util
import pygame
import numpy as np  
from scipy.spatial import distance
import gymnasium as gym
from gymnasium import spaces
import pygame_gui
from stable_baselines3 import PPO, DDPG
import csv 
from robot6 import Robot
from env6 import Environment 

# Here I implement an actual rotating robot actuator, instead of just a damped spring.

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
        
        self.space.add(self.env.anchor, self.env.anchor_shape, self.env.surface, self.env.surface_shape, self.env.spring)
        self.space.add(self.robot.body, self.robot.body_shape, 
                       self.robot.upper_arm, self.robot.upper_arm_shape, 
                       self.robot.lower_arm, self.robot.lower_arm_shape, 
                       self.robot.EE, self.robot.EE_shape, 
                       self.robot.actuator, self.robot.shoulder_pin, self.robot.elbow_pin, self.robot.wrist_pin)
        
        groove_end = (0, self.SIZE[1]-self.env.anchor.position[1]*2)
        surface_groove = pymunk.GrooveJoint(self.env.anchor, self.env.surface, (0, 0), groove_end, (0, 0))
        robot_body_groove = pymunk.GrooveJoint(self.env.anchor, self.robot.body, (0, 0), groove_end, (0, 0))
        robot_EE_groove = pymunk.GrooveJoint(self.env.anchor, self.robot.EE, (0, 0), groove_end, (0, 0))
        self.space.add(surface_groove, robot_body_groove, robot_EE_groove)
        
        self.steps = 0
        self.stable_steps = 0
        self.terminated = False
        self.truncated = False
        self.steps_per_episode = 2048
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        # # Lows and highs need to be modified acc to new stiffness and angle limits
        # obs_lows = np.array([0, -5000, -5000, 0, 0, 0, -5000, -5000, 0], dtype=np.float32) 
        # obs_highs = np.array([1000, 5000, 5000, 500, 1000, 1000, 5000, 5000, 1], dtype=np.float32)
        # self.observation_space = spaces.Box(low=obs_lows, high=obs_highs, shape=(9,))
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        self.total_reward = 0
        self.avg_reward = 0
        
        self.K_CONTROL = 'incremental'
        self.THETA_CONTROL = 'incremental'
        
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
        self.robot.do_action(action, noise=False, theta_control=self.THETA_CONTROL, k_control=self.K_CONTROL)
        self.space.step(1/self.FPS)
        self.steps += 1
        observation = self.get_observation()
        self.truncated = self.steps >= self.steps_per_episode
        self.anti_cheat()
        reward = self.reward = self.get_reward()
        return observation, reward, self.terminated, self.truncated, {}
    
    def update(self, action=None):
        if action is not None:
            self.robot.do_action(action, noise=False, theta_control=self.THETA_CONTROL, k_control=self.K_CONTROL)
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
        OBSERVATION_FUNCTION = 2
        self.robot.update_accelerations()
        self.env.update_accelerations()
        if self.env.surface.position.y - self.robot.EE.position.y < 30:
            self.robot.EE_contact = 1
        else:
            self.robot.EE_contact = 0
            
        match OBSERVATION_FUNCTION:
            case 1:
                # Does not output the current angle of the actuator, could be a problem
                observation = np.array([
                    self.robot.body.position.y,
                    self.robot.body.velocity.y,
                    self.robot.body_acceleration,
                    self.robot.actuator.rest_angle,
                    self.robot.actuator.stiffness,
                    self.robot.EE.position.y,
                    self.robot.EE.velocity.y,
                    self.robot.EE_acceleration,
                    self.robot.EE_contact
                ], dtype=np.float32)
            case 2:
                observation = np.array([
                    self.robot.body_acceleration,
                    self.robot.EE.position.y-self.robot.body.position.y,
                    self.robot.EE.velocity.y-self.robot.body.velocity.y,
                    self.robot.actuator.rest_angle,
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
        REWARD_FUNCTION = 2
        match REWARD_FUNCTION:
            case 1:
                reward = np.exp(-((self.robot.body.velocity.y / 25) ** 2))
            case 2:
                if self.robot.body.velocity.y < 50 and self.robot.EE_contact:
                    self.stable_steps += 1
                else:
                    self.stable_steps = 0  
                if self.stable_steps > 20:
                    reward = 1
                else:
                    reward = 0
        
        if self.out_of_bounds:
            reward = -100
        
        self.total_reward += reward
        self.avg_reward = self.total_reward/self.steps
        return reward
    
    def reset(self, seed=42):
        RANDOM_ON_RESET = True
        np.random.seed(seed)
        
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
        self.avg_reward = 0
        return self.get_observation(), {}
    
    def close(self):
        self.file.close()
        pygame.quit()
        sys.exit()
        
    def render(self):
        self.manager.update(1/self.FPS)
        self.screen.fill((255,255,255))
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
        text = f"Actuator rest angle: {self.robot.actuator.rest_angle}"
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
        
        text = "Robot MASS: "+ str(self.robot.body.mass)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 780))
        text = "Env Stiffness: "+ str(self.env.spring.stiffness)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 800))
        text = "Robot upper_arm angle: "+ str(self.robot.upper_arm.angle)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 820))
        text = "Robot lower_arm angle: "+ str(self.robot.lower_arm.angle)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 840))
        
        text = "Avg Reward per Step: "+ str(self.avg_reward)  
        text_surface = self.font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, (700, 860))
        
        
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
                 self.robot.actuator.rest_angle,
                 self.robot.actuator.stiffness,
                 self.env.surface.position.y,
                 self.env.surface.velocity.y,
                 self.env.surface_acceleration]
        self.writer.writerow(state)
    
    def add_gui_elements(self):
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
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.pause_button:
                    self.paused = not self.paused
                elif event.ui_element == self.reset_button:
                    seed = np.random.randint(0, 1000)
                    self.reset(seed=seed)
            self.manager.process_events(event)
            
    def mathematical_control(self):
        RETURN_ACTION = True
        
        alpha = 30
        m = self.robot.body.mass + self.robot.upper_arm.mass + self.robot.lower_arm.mass + self.robot.EE.mass
        g = self.env.gravity
        k = self.robot.actuator.stiffness
        rest_angle = self.robot.actuator.rest_angle
        theta = self.robot.upper_arm.angle - self.robot.lower_arm.angle
        d_theta = rest_angle - theta
        v_ee = self.robot.EE.velocity.y
        a_ee = self.robot.EE_acceleration
        
        theta_max = self.robot.ANGLE_LIMITS[1]
        theta_min = self.robot.ANGLE_LIMITS[0]
        k_max = self.robot.STIFFNESS_LIMITS[1]
        k_min = self.robot.STIFFNESS_LIMITS[0]
        
        upper_arm_length = ual = distance.euclidean(self.robot.p1, self.robot.p2)
        lower_arm_length = lal = distance.euclidean(self.robot.p2, self.robot.p3)
        
        l = self.robot.EE.position.y - self.robot.body.position.y
        
        if not RETURN_ACTION:
            if v_ee*a_ee>0 and abs(d_theta)>0.01:
                new_k = abs(m*g*ual*lal*np.sin(theta)/(l*d_theta))
                new_k = np.clip(new_k, k_min, k_max)
                self.robot.actuator.stiffness = new_k
            elif v_ee*a_ee<0 and abs(k)>10:
                new_d_theta = m*g*ual*lal*np.sin(theta)/(l*k)
                new_rest_angle = theta - new_d_theta
                new_rest_angle = np.clip(new_rest_angle, theta_min, theta_max)
                self.robot.actuator.rest_angle = new_rest_angle
        else:
            # only implementing incremental control for now
            normalized_k = normalized_theta = 0
            if v_ee*a_ee>0 and abs(d_theta)>0.01:
                new_k = abs(m*g*ual*lal*np.sin(theta)/(l*d_theta))
                new_k = np.clip(new_k, k_min, k_max)
                normalized_k = alpha*(new_k-k)/(k_max-k_min)
            elif v_ee*a_ee<0 and abs(k)>10:
                new_d_theta = m*g*ual*lal*np.sin(theta)/(l*k)
                new_rest_angle = theta - new_d_theta
                new_rest_angle = np.clip(new_rest_angle, theta_min, theta_max)
                normalized_theta = alpha*(new_rest_angle-rest_angle)/(theta_max-theta_min)
            action = np.array([normalized_theta, normalized_k])
            return action
        
    def random_control(self):
        action = np.random.uniform(-1, 1, size=(2,))
        return action
        
        
                
    
if __name__ == '__main__':
    robot = Robot()
    env = Environment()
    world = World(robot, env, mode='control')
    MODE = '-'
    
    if MODE == 'RL':
        model = PPO.load("models/PPO/model52")
        while True:
            action, _ = model.predict(world.get_observation())
            world.update(action=action)
            world.render()
    else:
        while True:
            action = world.mathematical_control()
            world.update(action=action)
            world.render()
    