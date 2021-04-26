from NeuroEvo.Genome.NodeGene import NodeGene
from pyro.distributions import *
from enum import Enum
import random
import torch
import numpy as np

class NodeType(Enum):
    Relu = 1
    Multiplication = 2
    Categorical = 3
    Gaussian = 4
    Dirichlet = 5

    # Returns a random node type according to a uniform prior
    @staticmethod
    def random(output = True):
        if output:
            return NodeType(random.randint(3,4))
        return NodeType(random.randint(1,5))

    def __str__(self):
         return self.name[0:3]

# Creates a node with a nodeNr
class AdvancedNodeGene(NodeGene):
    def __init__(self, nodeNr, layer = 0, output = False, input = False, outputtingTo = None, outputtingToClass = None, type=1, classCount = 1):
        super(AdvancedNodeGene, self).__init__(nodeNr, layer, output, input, outputtingTo)
        self.outputtingToClass = outputtingToClass
        self.type = NodeType(type)
        self.classCount = classCount
        self.inputs = []

    def function(self):
        if(self.input):
            output = self.inputs[0]
            self.inputs = []
            return output
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
        self.inputs = []
        return max(output, 0)

    # Returns the product of all inputs
    def multiply(self):
        output = 1
        for input in self.inputs:
            output *= input[0]
        self.inputs = []
        return output

    # Returns a Categorical distribution
    def categorical(self):
        inputs = torch.zeros((self.classCount, len(self.inputs[0][0])))
        for input in self.inputs:
            # Parameter index: input[1]
            # Parameter value: input[0]
            inputs[input[1]] = input[0]

        # Negative numbers are not allowed in categoricals
        # So we offset all values to positive by subtracting the smallest number, preserving the inputs ratios
        inputs -= torch.minimum(
            torch.min(inputs.clone(), dim=0).values,
            torch.zeros(len(torch.min(inputs.clone(), dim=0).values)))
        # Normalize inputs to distribution probabilities
        inputs = inputs / torch.max(torch.sum(inputs, dim=0), torch.ones(len(torch.sum(inputs, dim=0))))

        # If all inputs are 0, set the probabilities to be uniform TODO: check if this replaces the correct values
        sums = torch.sum(inputs, dim=0)
        for i, sum in enumerate(sums):
            if sum==0:
                inputs[:,i] = torch.ones(self.classCount)/self.classCount

        # Clear activation
        self.inputs = []
        return Categorical(inputs.T)

    # Returns a gaussian distribution
    def gaussian(self):
        inputs = torch.zeros(2, len(self.inputs[0][0]))

        for input in self.inputs:
            # Parameter index: input[1]
            # Parameter value: input[0]
            inputs[input[1]] = input[0]

        # Gaussians variance may not be negative or zero, so we truncate it to be above a small value
        inputs[1] = torch.maximum(
            inputs[1].clone(),
            torch.ones(len(inputs[1])) * 0.0000001)

        # Clear activation
        self.inputs = []

        return Normal(inputs[0], inputs[1])

    # Returns a dirichlet distribution
    def dirichlet(self):
        inputs = torch.zeros(self.classCount)
        for input in self.inputs:
            # Parameter index: input[1]
            # Parameter value: input[0]
            inputs[input[1]] = input[0]

        # The Dirichlet distribution may not receive parameter values smaller or equal to 0,
        # so we truncate the values to a small number
        for i in range(len(inputs)):
            inputs[i] = max(inputs[i], 0.0000001)

        self.inputs = []
        return Dirichlet(inputs)

    def linear(self):
        output = 0
        for input in self.inputs:
            output += input[0]
        self.inputs = []
        return output

    def __repr__(self):
        if(self.output):
            nodeType = "Output"
        elif(self.input):
            nodeType = "Input"
        else:
            nodeType = "Hidden"

        return "NodeNr: " + str(self.nodeNr) \
               + " Layer: " + str(self.layer) \
               + " Outputs to: " \
               + str(self.outputtingTo)  + " " \
               + "NodeType: " + nodeType + " "\
               + "Function: " + str(self.type)

    def __deepcopy__(self, memodict={}):
        return AdvancedNodeGene(self.nodeNr, self.layer, self.output, self.input,
                                outputtingTo= self.outputtingTo.copy(), type=self.type, classCount=self.classCount)
