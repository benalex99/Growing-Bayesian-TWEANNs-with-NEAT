import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuroEvo.NeuralNetwork.Layer import Layer

# Neural network class for running predictions on the GPU
class NeuralNetwork(nn.Module):

    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.inputSize = layers[0]
        self.outputSize = layers[len(layers) - 1]

        self.layers = []
        for weights, biases in layers:
            self.layers.append(Layer(weights.t(), biases))

    def randomInit(self, inputSize, outputSize, hiddenLayerSizes = []):
        hiddenLayerSizes.insert(0, inputSize)
        hiddenLayerSizes.append(outputSize)
        layers = []
        for i in range(len(hiddenLayerSizes) - 1):
            layers.append(Layer(hiddenLayerSizes[i], hiddenLayerSizes[i+1]))
        super(NeuralNetwork, self).__init__(layers)

    # Initialize a network from a Genome (using NEAT?)
    def fromGenome(self, genome):
        return

    # Calculates the output of the given input
    def forward(self, x):
        x = torch.tensor(x, dtype= torch.float)
        for layer in self.layers:
            x = F.relu(layer.forward(x))
        return x