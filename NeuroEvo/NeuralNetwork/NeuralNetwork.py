import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from blitz.modules import BayesianLinear

# Neural network class for running predictions on the GPU
class NeuralNetwork(nn.Module):

    def __init__(self, layers, allToAll):
        super(NeuralNetwork, self).__init__()
        self.cuda = torch.device('cpu')

        self.allToAll = allToAll
        if(not self.allToAll):
            if( layers == []):
                return
            self.inputSize = layers[0]
            self.outputSize = layers[len(layers) - 1]

            self.layers = []
            for weights, biases in layers:
                linearL = nn.Linear(1, 1)
                with torch.no_grad():
                    linearL.weight = torch.nn.Parameter(weights)
                    linearL.bias = torch.nn.Parameter(biases)
                linearL.to(self.cuda)
                self.layers.append(linearL)
        else:
            self.fromToLayers = []
            for fromLayer in layers:
                toLayers = []
                for weights, biases in fromLayer:
                    linearL = nn.Linear(1, 1)
                    # linearL = BayesianLinear(len(weights),len(weights[0]))
                    with torch.no_grad():
                        linearL.weight = torch.nn.Parameter(weights)
                        linearL.bias = torch.nn.Parameter(biases)
                    linearL.to(self.cuda)
                    toLayers.append(linearL)
                self.fromToLayers.append(toLayers)

            self.outputs = []
            self.outputs.append(torch.tensor([]))
            for i, fromLayer in enumerate(self.fromToLayers):
                if (len(fromLayer) == 0):
                    break
                self.outputs.append(torch.tensor(np.zeros(fromLayer[0].bias.size()), dtype=torch.float).to(self.cuda))

    # Calculates the output of the given input
    def forward(self, x):
        x = torch.tensor(x, dtype= torch.float, device= self.cuda)

        if not self.allToAll:
            for layer in self.layers:
                x = F.relu(layer(x))
            return x
        else:
            for output in self.outputs:
                output.fill_(0)
            self.outputs[0] = x

            for i, fromLayer in enumerate(self.fromToLayers):
                for i2, toLayer in enumerate(fromLayer):
                    self.outputs[i + i2 + 1] = self.outputs[i + i2 + 1] + (F.relu(toLayer(self.outputs[i])))
            return self.outputs[len(self.fromToLayers)-1]