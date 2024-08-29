import sys
import os
sys.path.append('worlds/world5')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
sys.path.append('worlds/world5')
sys.path.append('worlds/world6')
from robot5 import Robot
from env5 import Environment
from world5 import World
from stable_baselines3 import PPO, SAC, A2C

robot = Robot()
env = Environment()
world = World(robot, env, mode='RL')

ep_len = 3000
n_eps = 25

rewards = []
# model = PPO.load("models/PPO/model54")
model = SAC.load("models/SAC/model4")

for i in range(n_eps):
    obs = world.reset()
    for _ in range(ep_len):
        action, _ = model.predict(world.get_observation())
        # action = world.mathematical_control() 
        _, _, terminated, truncated, _ = world.update(action=action)
        if terminated or truncated:
            rewards.append(world.total_reward)
            break

print("Rewards:", rewards)
print("Average Reward:", sum(rewards) / len(rewards))