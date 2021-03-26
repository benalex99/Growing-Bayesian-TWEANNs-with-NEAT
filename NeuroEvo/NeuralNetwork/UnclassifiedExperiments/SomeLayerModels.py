
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(D_in, H)
        self.xor1 = nn.Linear(H,H)
        self.lin2 = nn.Linear(H, D_out)

    def forward(self, x):
        x = self.lin1(x)
        # print("x1: " + str(x))
        # x = self.xor1(x)
        # print("x2: " + str(x))
        return torch.sigmoid(self.lin2(x))

class XOR(nn.Module):
    def __init__(self, D_in, D_out):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(D_in, D_out)
        weights = []
        for inp in range(D_in):
            w = np.ones(D_out, dtype= float) * -1
            w[inp] = 1
            weights.append(w)
        with torch.no_grad():
            self.lin1.weight = nn.Parameter(torch.tensor(weights, dtype= torch.float))
        self.lin1.requires_grad_(False)

        self.weight = self.lin1.weight

    def forward(self, x):
        return F.relu(self.lin1(x))

class NonMaxUnlearn(nn.Module):
    def __init__(self, D_in, D_out):
        super(NonMaxUnlearn, self).__init__()
        self.xor = XOR(D_in, D_out)

    def forward(self, x):
        output, _ = torch.max(torch.abs(x), 1)

        maxes, _ = torch.max(self.xor(x), 1)
        mins, _ = torch.min(self.xor(x), 1)

        maxSigns = torch.sign(maxes)
        minSigns = torch.sign(mins)

        signs = maxSigns.int() | minSigns.int()
        result = output * signs
        return torch.reshape(result, (len(result),1))
