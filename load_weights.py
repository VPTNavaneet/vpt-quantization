import torch
import pickle

# Load and inspect the weights
weights = torch.load("rl-from-early-game-2x.weights")
for name, param in weights.items():
    print(name, param.shape)