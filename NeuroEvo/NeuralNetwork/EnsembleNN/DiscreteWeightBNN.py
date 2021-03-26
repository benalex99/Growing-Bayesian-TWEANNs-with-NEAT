
import torch
import torch.nn as nn
import torch.random as random
import torch.nn.functional as F


class DWBNN(nn.Module):
    def __init__(self, layers, weightCount = 1, device = "cpu"):
        super(DWBNN, self).__init__()
        self.device = device
        self.layers = []
        for h in layers:
            self.layers.append(DWBLayer(h[0], h[1], weightCount).to(torch.device(self.device)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = torch.tensor(x).to(torch.device(self.device)).float()
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x.cpu()

class DWBLayer(nn.Module):
    def __init__(self, D_in, D_out, weightCount = 1, device = "cpu"):
        super(DWBLayer, self).__init__()
        self.device = device
        self.initWeights(D_in,D_out, weightCount)

    def forward(self, x):
        self.sampleWeights()
        return F.relu(self.lin(x))

    def initWeights(self, D_in, D_out, weightCount):
        # initialize our discrete weight distributions with a single weight each
        self.weights = torch.rand((D_in, D_out, weightCount))
        self.lin = nn.Linear(D_in, D_out)

    def sampleWeights(self):
        # Sample from our discrete weight distributions
        for x, toNode in enumerate(self.weights):
            for y, fromNode in enumerate(toNode):
                self.lin.weight[y,x] = self.sampleWeight(fromNode)

    def sampleWeight(self, weights, N = 1):
        indice = torch.randint(0, len(weights), (N,1))
        indice = torch.tensor(indice)
        return weights[indice]