import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
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

file = open("models/SAC/model_no.txt", 'r')
model_no = int(file.read())
file.close()

model = SAC(
    policy='MlpPolicy',
    env=world,
    learning_rate=7.3e-4,
    buffer_size=8192,
    batch_size=256,
    ent_coef='auto',
    gamma=0.98,
    tau=0.02,
    train_freq=8,
    gradient_steps=8,
    learning_starts=10000,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-3, net_arch=[64, 64]),
    verbose=1,
    tensorboard_log='logs/'
)

try:
    model.learn(total_timesteps=2_000_000)
finally:
    model.save(f"models/SAC/model{model_no}")
    file = open("models/SAC/model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))