import sys
sys.path.append('worlds/world5')
from robot5 import Robot
from env5 import Environment
from world5 import World

import gymnasium as gym
import torch as th
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')
check_env(world)
world = Monitor(world)

file = open("models/DDPG/model_no.txt", 'r')
model_no = int(file.read())
file.close()

n_actions = world.action_space.shape[-1]
action_noise = NormalActionNoise(mean=th.zeros(n_actions), sigma=0.1 * th.ones(n_actions))

model = DDPG(
    policy="MlpPolicy",
    env=world,
    # learning_rate=1e-3,
    # buffer_size=10000,
    # learning_starts=50000,
    # batch_size=64,
    # tau=0.005,
    # gamma=0.9,
    # action_noise=action_noise,
    # policy_kwargs=dict(activation_fn=th.nn.ReLU),
    verbose=1,
    tensorboard_log="logs/"
)

try:
    model.learn(total_timesteps=500_000)
finally:
    model.save(f"models/DDPG/model{model_no}")
    file = open("models/DDPG/model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))