import numpy as np
import torch
import torch.nn as nn
from NeuroEvo.NeuralNetwork.ProbabilisticNN.PLayer import PLayer


# Neural network class for running predictions on the GPU
class PNN(nn.Module):

    def __init__(self, layers, allToAll):
        super(PNN, self).__init__()
        self.cuda = torch.device('cpu')

        self.allToAll = allToAll
        if(not self.allToAll):
            if( layers == []):
                return
            self.inputSize = layers[0]
            self.outputSize = layers[len(layers) - 1]

            self.layers = []
            for weights, biases in layers:
                self.layers.append(PLayer(self.cuda, weights.t(), biases))
        else:
            self.fromToLayers = []
            for fromLayer in layers:
                toLayers = []
                for weights, biases in fromLayer:
                    toLayers.append(PLayer(self.cuda, weights, biases))
                self.fromToLayers.append(toLayers)

            self.outputs = []
            self.outputs.append(torch.tensor([]))
            for i, fromLayer in enumerate(self.fromToLayers):
                if (len(fromLayer) == 0):
                    break
                self.outputs.append(torch.tensor(np.zeros(len(fromLayer[0].bias)), dtype=torch.float).to(self.cuda))

    @classmethod
    def randomInit(self, inputSize, outputSize, hiddenLayerSizes = []):
        hiddenLayerSizes.insert(0, inputSize)
        hiddenLayerSizes.append(outputSize)
        layers = []
        for i in range(len(hiddenLayerSizes) - 1):
            layers.append(PLayer(hiddenLayerSizes[i], hiddenLayerSizes[i+1]))
        super(PNN, self).__init__(layers)

    # Calculates the output of the given input
    def forward(self, x):
        x = torch.tensor(x, dtype= torch.float, device= self.cuda)

        if not self.allToAll:
            for layer in self.layers:
                activation = layer.forward(x)
                prob = torch.bernoulli(activation)
                x = prob * activation + (1 - prob)
            return x
        else:
            for output in self.outputs:
                output.fill_(0)
            self.outputs[0] = x

            for i, fromLayer in enumerate(self.fromToLayers):
                for i2,toLayer in enumerate(fromLayer):
                    activation = toLayer.forward(self.outputs[i])
                    prob = torch.bernoulli(activation)
                    self.outputs[i + i2 + 1] = self.outputs[i + i2 + 1] * (prob * activation + (1 - prob))
            return self.outputs[len(self.fromToLayers)-1]