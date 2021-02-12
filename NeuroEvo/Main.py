import Trainer
import torch.nn as nn
from NeuralNetwork.NeuralNetwork import NeuralNetwork
import numpy as np
import torch


# Choose an environment and optimizer, run trainer
# example nn with layers 5,2,3
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome

layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers)

gg = NEATGenome()
gg.addNode()
gg.addNode()
gg.addNode()
gg.addNode()
gg.addEdge()

print(gg.edges)

print(nn.forward([1,2,3,4,5]))

