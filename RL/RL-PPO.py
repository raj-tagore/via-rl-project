from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import sys
sys.path.append('worlds/world5')
sys.path.append('worlds/world6')
from robot5 import Robot
from env5 import Environment
from world5 import World
# from robot6 import Robot
# from env6 import Environment
# from world6 import World
import torch.nn as nn

robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')
check_env(world)
world = Monitor(world)
# world = DummyVecEnv([lambda: world])
# world = VecNormalize(world, norm_obs=True)

file = open("models/PPO/model_no.txt", 'r')
model_no = int(file.read())
file.close()

# Set up the hyperparameters
params = {
    'n_steps': 2048,
    'batch_size': 64,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'n_epochs': 10,
    'ent_coef': 0.0,
    'learning_rate': 2.5e-4,
    'clip_range': 0.2,
    'verbose': 1,
    'normalize_advantage': True,          
    'tensorboard_log': 'logs/',
}

model = PPO('MlpPolicy', world, **params)

try:
    model.learn(total_timesteps=2_000_000)
finally:
    model.save(f"models/PPO/model{model_no}")
    file = open("models/PPO/model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))