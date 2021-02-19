import torch
from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuroEvo.Optimizers.NEAT.NEATGenome import *
from NeuroEvo.Optimizers.NEAT.NEAT import *

layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers, False)


print("make genome")

gg = NEATGenome(5, 2)

optim = NEAT()
gg = optim.testRun(gg,10,20,50)

gg.visualize()
print("transform to pytorch nn")
nn = gg.toNN()
print("Output: " + str(nn.forward([1,2,3,4,5]).tolist()))
