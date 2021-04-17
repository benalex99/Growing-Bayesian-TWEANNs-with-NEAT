from NeuroEvo.Genome.NodeGene import NodeGene
from pyro.distributions import *
from enum import Enum
import random

class NodeType(Enum):
    Relu = 1
    Multiplication = 2
    Categorical = 3
    Gaussian = 4
    Dirichlet = 5

    # Returns a random node type according to a uniform prior
    @staticmethod
    def random():
        return NodeType(random.randint(1,5))

# Creates a node with a nodeNr
class AdvancedNodeGene(NodeGene):
    def __init__(self, nodeNr, layer = 0, output = False, input = False, outputtingTo = None, type=1):
        super(AdvancedNodeGene, self).__init__(nodeNr, layer, output, input, outputtingTo)
        self.type = NodeType(type)
        self.inputs = []

    def function(self):
        if(self.type == NodeType.Relu):
            return self.relu()
        if(self.type == NodeType.Multiplication):
            return self.multiply()
        if(self.type == NodeType.Categorical):
            return self.categorical()
        if(self.type == NodeType.Gaussian):
            return self.gaussian()
        if(self.type == NodeType.Dirichlet):
            return self.dirichlet()

    # Returns the Relu of the sum of inputs
    def relu(self):
        output = 0
        for input in self.inputs:
            output += input[0]
        return max(0, output)

    # Returns the product of all inputs
    def multiply(self):
        output = 1
        for input in self.inputs:
            output *= input[0]
        return output

    # Returns a Categorical distribution
    def categorical(self):
        inputs = torch.zeros(len(self.inputs))
        for input in self.inputs:
            # Reshuffle the inputs from layer ordering to actual class allocations
            inputs[input[1]] = input[0]
            # Normalize inputs to distribution probabilities
            inputs = inputs / sum(inputs)
        return Categorical(inputs)

    # Returns a gaussian distribution
    def gaussian(self):
        inputs = []
        for _ in self.inputs:
            inputs.append(0)
        for input in self.inputs:
            # Class (input[1]) probability (input[0])
            inputs[input[1]] = input[0]
        return Normal(inputs[0], inputs[1])

    # Returns a dirichlet distribution
    def dirichlet(self):
        inputs = torch.zeros(len(self.inputs))
        for input in self.inputs:
            # Reshuffle the inputs from layer ordering to actual class allocations
            inputs[input[1]] = input[0]
        return Dirichlet(inputs)
