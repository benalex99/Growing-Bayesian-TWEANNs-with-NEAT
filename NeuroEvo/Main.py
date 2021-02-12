import Trainer
import torch.nn as nn
from NeuralNetwork.NeuralNetwork import NeuralNetwork
import numpy as np
import torch


# Choose an environment and optimizer, run trainer
# example nn with layers 5,2,3
layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers)

print(nn.forward([1,2,3,4,5]))
