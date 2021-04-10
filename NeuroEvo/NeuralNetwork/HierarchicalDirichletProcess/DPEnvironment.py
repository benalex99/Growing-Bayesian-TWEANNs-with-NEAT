from pyro.distributions import *
import torch
import numpy as np

class Env():
    def CategoricalOfCategoricals(self, N= 200, classCount = 5, featureCount = 10):
        self.categoricals = []
        for i in range(classCount):
            self.categoricals.append(Categorical(Dirichlet(1/featureCount * torch.ones(featureCount)).sample([1])))

        self.classProbs = Dirichlet(1/classCount * torch.ones(classCount)).sample([1])

        data = []
        for _ in range(N):
            data.append(self.categoricals[Categorical(self.classProbs).sample([1])[0]].sample([1]))
        data = torch.Tensor(data).flatten().int()
        return data

class NestedCategorical():
    """
    A tree of nested categorical distributions with random
    """
    def __init__(self, depth, width):
        self.classProbs = Dirichlet(1/width * torch.ones(width)).sample([1])[0]
        self.children = []
        if(depth > 1):
            for _ in range(width):
                self.children.append(NestedCategorical(depth-1, width))

    def sample(self, N):
        data = []
        if(N > 0):
            sampleCount = Multinomial(N, self.classProbs).sample([1])[0].int()
            if(len(self.children) > 0):
                for i, child in enumerate(self.children):
                        data.append(child.sample(sampleCount[i].item()))
                else:
                    return sampleCount
        return data

class NestedCategoricalWithBias():
    """
    A tree of nested categorical distributions with random child probabilities
    """
    def __init__(self, depth, width, bias):
        self.bias = bias
        self.classProbs = Dirichlet(1/width * torch.ones(width)).sample([1])[0]
        self.children = []
        if(depth > 1):
            for _ in range(width):
                self.children.append(NestedCategoricalWithBias(depth-1, width, bias))

    def sample(self, N):
        data = []
        if(N+self.bias > 0):
            sampleCount = Multinomial(N, self.classProbs).sample([1])[0].int()
            if(len(self.children) > 0):
                for i, child in enumerate(self.children):
                    sample = child.sample(sampleCount[i].item())
                    if(len(sample) > 0):
                        data.append(np.array(sample))
            else:
                return sampleCount.numpy()
        if(np.array(data).ndim == 1):
            return np.array(data)
        return np.array(data).sum(0)

class hierarchicalDirichletData():
    def __init__(self, depth, width):
        phi = Dirichlet(1/width * torch.ones(width)).sample([1])[0]
        pseudoCounts = Multinomial(1000, phi).sample([1])[0]
        
        self.upperDirichlet = Dirichlet(pseudoCounts)


    def sample(self, N):
