import torch
from stable_baselines3 import PPO

model = PPO.load("models/PPO/model50")
policy_network = model.policy

for name, param in policy_network.named_parameters():
    if "weight" in name:
        print(f"Layer: {name}, Weights:\n{param.data}\n")

# Access specific layers and analyze the input layer
input_layer_weights = policy_network.mlp_extractor.policy_net[0].weight  # Assuming this is the first layer
print("Input Layer Weights:\n", input_layer_weights.data)

# To calculate the average magnitude of weights per input feature
average_weights = torch.mean(torch.abs(input_layer_weights), dim=0)
print("Average Magnitude of Weights for Each Input Feature:\n", average_weights)

