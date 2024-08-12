from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
import sys
sys.path.append('worlds/world5')
from robot5 import Robot
from env5 import Environment
from world5 import World

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
    'learning_rate': 1e-4,      # Learning rate for the optimizer
    'n_steps': 512,             # Number of steps to run for each environment per update
    'batch_size': 128,          # Mini-batch size for each gradient update
    'n_epochs': 2,              # Number of optimization epochs to perform
    'gamma': 0.9,               # Discount factor for future rewards
    'gae_lambda': 0.95,         # GAE lambda parameter
    'ent_coef': 0.005,          # Coefficient for entropy bonus
    'verbose': 0,               # Verbosity level for logging
    'tensorboard_log': 'logs/', # Directory to save logs
}

model = PPO('MlpPolicy', world, **params)

try:
    model.learn(total_timesteps=5_000_000)
finally:
    model.save(f"models/PPO/model{model_no}")
    file = open("models/PPO/model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))