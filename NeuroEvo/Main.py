import torch
<<<<<<< Updated upstream


# Choose an environment and optimizer, run trainer
# example nn with layers 5,2,3
from NeuroEvo.Genome.Genome import Genome
from Optimizers.NEAT.NEATGenome import NEATGenome
=======
from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuroEvo.Optimizers.NEAT.NEATGenome import *
>>>>>>> Stashed changes

layers = [[torch.ones(5,2),torch.ones(2)], [torch.ones(2,3),torch.ones(3)]]

nn = NeuralNetwork(layers)


print("go")

gg = NEATGenome()
gg.addNode()
gg.addNode()
gg.addNode()
gg.addNode()
gg.addEdge()
<<<<<<< Updated upstream
gg.tweakWeight(0.1)
=======
gg.addEdge()
gg.addEdge()

nn = gg.toNN()
>>>>>>> Stashed changes

print(gg.edges[0].weight)

print(nn.forward([1,2,3,4,5]))

