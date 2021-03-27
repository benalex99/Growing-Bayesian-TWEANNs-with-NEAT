from NeuroEvo.Genome.Genome import Genome as Genome
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class GrandmaNeuronOptimizer():

    @staticmethod
    def train(xData, yData):
        # yo, what is the difference between xData and yData? xData maps to a neuron, which then maps back out into y


        concepts = []

        # Test if data is already in network
        isInNetwork = yData == decoder(encoder(xData))

        if not isInNetwork:
            # Map xData to one hot vector
            encoder
            # Map one hot vector yData