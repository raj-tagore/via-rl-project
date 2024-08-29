from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
import sys
sys.path.append('worlds/world5')
from robot5 import Robot
from env5 import Environment
from world5 import World
import numpy as np

robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')
check_env(world)
world = Monitor(world)

file = open("models/TD3/model_no.txt", 'r')
model_no = int(file.read())
file.close()

n_actions = world.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MlpPolicy", world, action_noise=action_noise, verbose=1)

try:
    model.learn(total_timesteps=2_000_000)
finally:
    model.save(f"models/TD3/model{model_no}")
    file = open("models/TD3/model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))