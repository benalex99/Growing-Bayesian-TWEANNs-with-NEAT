import math
import torch
import torch.nn as nn
import numpy as np

class Layer(nn.Module):

    def __init__(self, weights = np.array([[]]), biases = []):
        super().__init__()
        self.size_in, self.size_out = weights.size()[0], weights.size()[1]
        self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        self.bias = nn.Parameter(biases)

    @classmethod
    def randomly(self, size_in, size_out):
        self.size_in, self.size_out = size_in, size_out
        weights = torch.Tensor(size_out, size_in)
        weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        bias = torch.Tensor(size_out)
        bias = nn.Parameter(bias)

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5)) # weight init
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        super(Layer, self).__init__(weights, bias)

    def forward(self, x):
        w_times_x = torch.matmul(x.float(), self.weights.t())
        return torch.add(w_times_x, self.bias) # w times x + b