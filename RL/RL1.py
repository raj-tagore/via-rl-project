import pymunk
import pymunk.pygame_util
import pygame
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

class Env_1D_SpringJoint(gym.Env):
    
    pygame.init()
    metadata = {"render_modes": ["human", "graphs", "both"], "render_fps": 4}
    
    def __init__(self, gravity, surface_properties, render_mode=None):
        
        # surface_properties = (rest_length, stiffness, damping)
        
        super(Env_1D_SpringJoint, self).__init__()
        
        self.action_space = spaces.Box(-1, 1, (2,))
        self.observation_space = spaces.Dict({
            'feet_contact': spaces.MultiBinary((1,)),
            'self_actuator': spaces.Box(-1000, 1000, (2,)),
            'self_body': spaces.Box(-1000, 1000, (2,))
        })
        
        
        self.clock = pygame.time.Clock()
        
        self.space = pymunk.Space()
        self.space.gravity = (0, gravity)
        
        self.anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.anchor_shape = pymunk.Circle(self.anchor, 5)
        self.anchor.position = (150, 500)
        self.space.add(self.anchor, self.anchor_shape)
        
        self.surface = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 5))
        self.surface.position = (150, 400)
        self.surface_shape = pymunk.Circle(self.surface, 5)
        self.space.add(self.surface, self.surface_shape)
        
        self.spring1 = pymunk.DampedSpring(self.anchor, self.surface, (0, 0), (0, 0), surface_properties[0], surface_properties[1], surface_properties[2])
        self.space.add(self.spring1)
        
        self.surface_groove = pymunk.GrooveJoint(self.anchor, self.surface, (0, 0), (0, -500), (0, 0))
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
        self.stable_steps = 0
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
    
    def step(self, action):
        robot_actuator_position = (action[0]+1)*200
        robot_actuator_impedance = (action[1]+1)*100
        
        self.robot_actuator.stiffness = robot_actuator_impedance
        self.robot_actuator.rest_length = robot_actuator_position
        self.space.step(1/60)
        
        self.truncated = False if self.steps < 10000 else True
        self.terminated = False
        self.steps += 1
        
        if self.robot_body.velocity.y<10:
            self.stable_steps += 1
        else:
            self.stable_steps = 0
        
        if self.stable_steps>2000:
            self.terminated = True
        
        return self._get_state(), self._get_reward(), self.terminated, self.truncated, {}
    
    def reset(self, seed=42):
        
        np.random.seed(seed)
        surface_properties = (np.random.uniform(50, 150), np.random.uniform(200, 1000), 0)
        gravity = np.random.uniform(500, 1000)
        
        self.anchor.position = (150, 500)
        self.surface.position = (150, 400)
        self.robot_body.position = (150, 100)
        self.robot_EE.position = (150, 200)
        self.spring1 = pymunk.DampedSpring(self.anchor, self.surface, (0, 0), (0, 0), surface_properties[0], surface_properties[1], surface_properties[2])
        self.space.gravity = (0, gravity)
        self.steps = 0
        
        return self._get_state(), {}
    
    def render(self, mode="human"):
        if mode == "human":
            self.screen = pygame.display.set_mode((300, 600))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.screen.fill((255, 255, 255))
            self.space.debug_draw(self.draw_options)
            pygame.display.update()
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close(mode)
        elif mode == "graphs":
            pass
        else:
            pass
    
    def close(self, mode):
        pygame.quit()
        sys.exit()
    
    def _get_reward(self):
        k = 1/1000
        if self.terminated:
            return 1000
        return k*self.robot_body.velocity.y**2
    
    def _get_state(self):
        obs_dict = {
            'feet_contact': np.array([1],dtype=np.int8) if self.surface.position.y - self.robot_EE.position.y < 12 else np.array([0],dtype=np.int8),
            'self_actuator': np.array([self.robot_actuator.stiffness, self.robot_actuator.rest_length], dtype=np.float32),
            'self_body': np.array([self.robot_body.position.y, self.robot_body.velocity.y], dtype=np.float32),
        }
        
        return obs_dict
        
if __name__ == '__main__':
    env = Env_1D_SpringJoint(900, (100, 100, 0), "human")
    check_env(env)
    model = PPO('MultiInputPolicy', env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("models/pilot")
    model = PPO.load("models/pilot")
    for _ in range(1000):
        action, _states = model.predict(env._get_state(), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs = env.reset()
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    env.close()
    
    
        
        
        