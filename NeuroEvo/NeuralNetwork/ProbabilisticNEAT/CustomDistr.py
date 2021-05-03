

from pyro.distributions import Distribution, constraints
import torch
import math
import sys

class ReluDistr(Distribution):
    def __init__(self, inputs, supportUpperBound = sys.float_info.max, zeroProbEstimateDivisor = sys.float_info.max):
        self.supportUpperBound = supportUpperBound
        self.zeroProbEstimateDivisor = zeroProbEstimateDivisor
        self.inputs = inputs
        self.support = constraints.real
        pass

    def sample(self, sample_shape=torch.Size()):
        output = []
        if(sample_shape==torch.Size()):
            output = torch.relu(self.inputs)
        else:
            for _ in range(sample_shape[0]):
                output.append(torch.relu(self.inputs))
        self.inputs = []
        return output

    def log_prob(self, x, *args, **kwargs):
        logPs = torch.ones(x.shape, device=torch.device('cuda')) * self.zeroProbEstimateDivisor

        logPs[x == 0] = math.log(0.5)
        logPs[x > 0] = math.log(0.5/self.supportUpperBound)

        return logPs

class LeakyReluDistr(Distribution):
    def __init__(self, inputs, supportLowerBound = sys.float_info.min/16,
                 supportUpperBound = sys.float_info.max/16,
                 zeroProbEstimateDivisor = sys.float_info.max):
        self.supportLowerBound = supportLowerBound
        self.supportUpperBound = supportUpperBound
        self.zeroProbEstimateDivisor = zeroProbEstimateDivisor
        self.inputs = inputs
        self.support = constraints.real
        pass

    def sample(self, sample_shape=torch.Size()):
        output = []
        if(sample_shape==torch.Size()):
            output = torch.relu(self.inputs)
        else:
            for _ in range(sample_shape[0]):
                output.append(torch.relu(self.inputs))
        self.inputs = []
        return output

    def log_prob(self, x, *args, **kwargs):
        logPs = torch.ones(x.shape, device=torch.device('cuda')) * math.log(1/self.supportUpperBound - self.supportLowerBound)
        return logPs

class Identity(Distribution):
    def __init__(self, inputs, supportLowerBound = sys.float_info.min/16,
                 supportUpperBound = sys.float_info.max/16,
                 zeroProbEstimateDivisor = sys.float_info.max):
        self.supportLowerBound = supportLowerBound
        self.supportUpperBound = supportUpperBound
        self.inputs = inputs
        self.support = constraints.real
        pass

    def sample(self, sample_shape=torch.Size()):
        output = []
        if(sample_shape==torch.Size()):
            output = self.inputs
        else:
            for _ in range(sample_shape[0]):
                output.append(self.inputs)
        self.inputs = []
        return output

    def log_prob(self, x, *args, **kwargs):
        logPs = torch.ones(x.shape, device=torch.device('cuda')) * math.log(1/self.supportUpperBound - self.supportLowerBound)
        return logPs
