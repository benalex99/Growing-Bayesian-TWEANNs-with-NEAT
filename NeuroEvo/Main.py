import torch


# Choose an environment and optimizer, run trainer
# example nn with layers 5,2,3
from NeuroEvo.Genome.Genome import Genome
from Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuroEvo.Optimizers.NEAT.NEATGenome import *

layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers, False)


print("go")

gg = NEATGenome(5, 2)
print("nodes")
for i in range(0):
    gg.addNode()
print("edges")
for i in range(10):
    gg.addEdge()
gg.tweakWeight(0.1)

print("toNN")
nn = gg.toNN()

print(nn.forward([1,2,3,4,5]))

