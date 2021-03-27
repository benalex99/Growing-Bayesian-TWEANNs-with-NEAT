from NeuroEvo.Genome.Genome import Genome as Genome
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, D_in, D_out):
        super(Encoder, self).__init__()
        self.encoding1 = nn.Linear(D_in, 128)
        self.encoding2 = nn.Linear(128, 64)
        self.concepts = nn.Linear(64,1)

    def forward(self, x):
        x = torch.relu(self.encoding1(x))
        x = torch.relu(self.encoding2(x))
        x = torch.sigmoid(self.concepts(x))
        return x

class Decoder(nn.Module):
    def __init__(self, D_in, D_out):
        super(Decoder, self).__init__()
        self.concepts = nn.Linear(1, 64)
        self.decoding1 = nn.Linear(64, 128)
        self.decoding2 = nn.Linear(128, D_out)

    def forward(self, x):
        x = torch.relu(self.concepts(x))
        x = torch.relu(self.decoding1(x))
        x = torch.relu(self.decoding2(x))
        return x

class grandmaNeuronBrain():
    def __init__(self, inputSize, outputSize):
        self.encoder = Encoder(inputSize, 1)
        self.decoder = Decoder(1, outputSize)
        self.concepts = 0

    def learnConcept(self, x, y):
