from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import sys
sys.path.append('worlds/world5')
from robot5 import Robot
from env5 import Environment
from world5 import World
import torch.nn as nn

robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')

check_env(world)
world = Monitor(world)

model_path = "models/PPO/model37"

model = PPO.load(model_path, world)

try:
    model.learn(total_timesteps=10_000_000)
finally:
    model.save(model_path)    