
import torch
import torch.nn as nn



class Model(nn.Module):
    def __init__(self, layers, device = "cpu"):
        super(Model, self).__init__()
        self.device = device
        self.layers = []
        for h in layers:
            self.layers.append(nn.Linear(h[0], h[1]).to(torch.device(self.device)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = torch.tensor(x).float().to(torch.device(self.device))
        for i, layer in enumerate(self.layers):
            if(i < len(self.layers) - 1):
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        return x.cpu()