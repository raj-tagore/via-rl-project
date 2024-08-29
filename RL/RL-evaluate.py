import sys
import os
sys.path.append('worlds/world5')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from robot5 import Robot
from env5 import Environment
from world5 import World

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

model = PPO.load("models/PPO/model30")
robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')

mean_reward, std_reward = evaluate_policy(model, world, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward} +/- {std_reward}")