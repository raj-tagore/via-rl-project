from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from worlds.world4 import Robot, Environment, World

robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')
check_env(world)

file = open("model_no.txt", 'r')
model_no = int(file.read())
file.close()

model = PPO('MlpPolicy', world, learning_rate=0.0003, n_steps=8192, batch_size=256, n_epochs=10, verbose=1)
try:
    model.learn(total_timesteps=1_000_000)
finally:
    model.save(f"models/model{model_no}")
    file = open("model_no.txt", 'w')
    model_no += 1
    file.write(str(model_no))



"""
Notes:


""" 