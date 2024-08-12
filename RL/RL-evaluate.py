from stable_baselines3.common.evaluation import evaluate_policy
from worlds.world4 import Robot, Environment, World
from stable_baselines3 import PPO

model = PPO.load("models/model6")
robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')

mean_reward, std_reward = evaluate_policy(model, world, n_eval_episodes=10, render=False)
print(f"Mean reward: {mean_reward} +/- {std_reward}")