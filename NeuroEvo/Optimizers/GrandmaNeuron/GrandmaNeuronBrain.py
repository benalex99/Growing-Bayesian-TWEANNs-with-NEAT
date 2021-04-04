from NeuroEvo.Genome.Genome import Genome as Genome
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GenomeNN(nn.Module):
    def __init__(self, D_in, D_out):
        super(GenomeNN, self).__init__()
        self.genome = Genome(D_in, D_out)
        self.model = self.genome.toNN()

    def forward(self, x):
        return self.model(x)

    # increase input size
    def increaseInput(self):
        self.genome.increaseInput()

    # increase output size
    def increaseOutput(self):
        self.genome.increaseOutput()

    # Adds an output node and a grandma neuron that connects to it with weight 1
    # Optimizes the grandma neuron in a top down or bottom up fashion.
    # TODO: implement
    def addGrandmaNeuronOutput(self):
        pass

    def addGrandmaNeuronInput(self):
        pass


# Test abstraction vs practicality control parameterization
# Have neurons connections created and tuned layer per layer.
# For abstract preference we start at the highest layer and tune. If we still have error we proceed to lower layers.
# For practical preference we start at the lowest layer and tune. If we still have error we proceed to higher layers.
# Using lower level inputs is equivalent to a more detailed hypothesis
# while using more abstract inputs is a less detailed hypothesis(?)
# TODO: Implement the above
# TODO: Test how this the bottom up vs top down approach result in different network structures.
# TODO: Amount of parameters (total connections)? Layers? Layer width?
# TODO: What does changing the order of the training data do?
# TODO: WHERE TO GET TRAINING DATA FROM??? Image data? Generative bayesian model? should be hierarchical... Random binary data!(?)
# TODO: Whats my research question? What is the influence abstraction in a growing network?
# When learning a new datapoint, if the prediction is already correct, don't add a new neuron.
# If the one hot encoding is partially correct, but not fully,
# use a threshold to determine whether the old weights should be adapted or a new node added.
class grandmaNeuronBrain():
    def __init__(self, inputSize, outputSize):
        self.encoder = GenomeNN(inputSize, 0)
        self.decoder = GenomeNN(0, outputSize)
        self.concepts = 0

    def learnConcept(self, x, y):
