import torch
from torch.utils.tensorboard import SummaryWriter

# Choose an environment and optimizer, run trainer
# example nn with layers 5,2,3
from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuroEvo.Optimizers.NEAT.NEATGenome import *

layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers, False)


print("make genome")

gg = NEATGenome(5, 2)
for i in range(10):
    gg.mutate()
gg.visualize()
print("transform to pytorch nn")
nn = gg.toNN()
print("Output: " + str(nn.forward([1,2,3,4,5]).tolist()))
