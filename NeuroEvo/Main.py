import torch


# Choose an environment and optimizer, run trainer
# example nn with layers 5,2,3
from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuroEvo.Optimizers.NEAT.NEATGenome import *

layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers, False)


print("make genome \n")

gg = NEATGenome(5, 2)
for i in range(20):
    gg.mutate()

print("transform to pytorch nn \n")
nn = gg.toNN()

print(nn.forward([1,2,3,4,5]))

