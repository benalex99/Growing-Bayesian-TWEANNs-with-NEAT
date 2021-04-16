from NeuroEvo.Genome.NodeGene import NodeGene
from pyro.distributions import *



# Creates a node with a nodeNr
class AdvancedNodeGene(NodeGene):
    def __init__(self, nodeNr, layer = 0, output = False, input = False, outputtingTo = None, type="Relu"):
        super(AdvancedNodeGene, self).__init__(nodeNr, layer, output, input, outputtingTo)
        self.type = type
        self.inputs = []

    def function(self):
        if(self.type == "Relu"):
            return self.relu()
        if(self.type == "Multiplication"):
            return self.multiply()
        if(self.type =="Categorical"):
            return self.categorical()
        if(self.type == "Gaussian"):
            return self.gaussian()
        if(self.type == "Dirichlet"):
            return self.dirichlet()

    # TODO: This is not a distribution. We should not sample for deterministic functions
    def relu(self):
        output = 0
        for input in self.inputs:
            output += input[0]
        return max(0, output)

    def multiply(self):
        output = 1
        for input in self.inputs:
            output *= input
        return output

    # TODO: Reshuffle inputs according to input and output indexing
    def categorical(self):
        inputs = []
        for _ in self.inputs:
            inputs.append(0)
        for input in self.inputs:
            # Class (input[1]) probability (input[0])
            inputs[input[1]] = input[0]
        return Categorical(inputs)

    # TODO: Reshuffle inputs according to input and output indexing
    def gaussian(self):
        inputs = []
        for _ in self.inputs:
            inputs.append(0)
        for input in self.inputs:
            # Class (input[1]) probability (input[0])
            inputs[input[1]] = input[0]
        return Normal(inputs[0], input[1])

    # TODO: Reshuffle inputs according to input and output indexing
    def dirichlet(self):
        inputs = []
        for _ in self.inputs:
            inputs.append(0)
        for input in self.inputs:
            # Class (input[1]) probability (input[0])
            inputs[input[1]] = input[0]
        return Dirichlet(inputs)
