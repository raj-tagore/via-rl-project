from stable_baselines3 import A2C
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

file = open("models/A2C/model_no.txt", 'r')
model_no = int(file.read())
file.close()

model = A2C(
    policy='MlpPolicy',
    env=world,
    learning_rate=3e-4, 
    n_steps=64,
    gamma=0.99,
    gae_lambda=0.9,
    ent_coef=0.0,
    vf_coef=0.4,
    max_grad_norm=0.5,
    use_rms_prop=True,
    normalize_advantage=False,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-2, ortho_init=False),
    verbose=1,
    tensorboard_log='logs/'
)

try:
    model.learn(total_timesteps=2_000_000)
finally:
    model.save(f"models/A2C/model{model_no}")
    file = open("models/A2C/model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))