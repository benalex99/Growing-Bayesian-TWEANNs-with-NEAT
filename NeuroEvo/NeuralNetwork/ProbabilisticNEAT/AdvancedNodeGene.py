from NeuroEvo.Genome.NodeGene import NodeGene
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.CustomDistr import ReluDistr, LeakyReluDistr, Identity
from pyro.distributions import *
from enum import Enum
import random
import torch
import numpy as np

class NodeType(Enum):
    Multiplication = 1
    Relu = 2
    Categorical = 3
    Gaussian = 4
    Dirichlet = 5
    Sum = 6

    # Returns a random node type according to a uniform prior
    @staticmethod
    def random(output = False):
        if output:
            return NodeType(random.randint(3,4))
        return NodeType(random.randint(2,5))

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
        self.edges = []

    def function(self):
        if(self.input):
            output = self.inputs[0]
            self.inputs = []
            return output
        if(self.output):
            return self.sum()
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
        output = []
        for input in self.inputs:
            output.append(input[0])
        output = torch.stack(output, dim=0)
        output = torch.sum(output, dim= 0)
        # output = torch.relu(output)
        self.inputs = []
        # return output
        return ReluDistr(output)

    # Returns the product of all inputs
    def multiply(self):
        output = 1
        for input in self.inputs:
            output *= input[0]
        self.inputs = []
        return output

    # Returns a Categorical distribution
    def categorical(self):
        # Assign inputs to parameters
        inputs = torch.zeros((self.classCount, len(self.inputs[0][0])), device=torch.device('cuda'))
        for input in self.inputs:
            # Parameter index: input[1]
            # Parameter value: input[0]
            inputs[input[1]] = input[0]

        # Clear activation
        self.inputs = []

        # Effectively a categorical.
        # Multinomial is used because it outputs hot vectors. These are in an easier representation for the network.
        return Multinomial(1, logits=inputs.T)

    # Returns a gaussian distribution
    def gaussian(self):
        # Assign inputs to parameters
        inputs = torch.ones(2, len(self.inputs[0][0]), device=torch.device('cuda'))
        for input in self.inputs:
            # Parameter index: input[1] (index of 0 is mean, index of 1 is variance)
            # Parameter value: input[0]
            inputs[input[1]] = input[0]

        # Gaussians variance may not be negative or zero, so we truncate it to be above a small value
        inputs[1][inputs[1]<=0] = 0.0000001

        # Clear activation
        self.inputs = []
        return Normal(inputs[0], inputs[1])

    # Returns a dirichlet distribution
    def dirichlet(self):
        # Assign inputs to parameters
        inputs = torch.zeros((self.classCount, len(self.inputs[0][0])), device=torch.device('cuda'))
        for input in self.inputs:
            # Parameter class: input[1]
            # Parameter value: input[0]
            inputs[input[1]] = input[0]

        # The Dirichlet distribution may not receive parameter values smaller or equal to 0,
        # so we truncate the values to a small number
        inputs[inputs<=0] = 0.0000001

        self.inputs = []
        return Dirichlet(inputs.T)

    def sum(self):
        output = []
        for input in self.inputs:
            output.append(input[0])
        output = torch.stack(output, dim=0)
        output = torch.sum(output, dim= 0)
        self.inputs = []
        return Identity(output)

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

    def toData(self):
        return [self.nodeNr, self.layer, self.output, self.input, self.outputtingTo, self.type, self.classCount]

    @staticmethod
    def fromData(data):
        return AdvancedNodeGene(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
