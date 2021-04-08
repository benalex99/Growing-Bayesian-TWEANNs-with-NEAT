from NeuroEvo.Genome import Genome, NodeGene, ConnectionGene
import torch


class DirichletGenome(Genome.Genome):
    def __init__(self, inputSize, outputSize):
        super().__init__(inputSize, outputSize)

    def forward(self, input):
        layers = self.getLayers()

        for layer in layers:
            for node in layer:
                node.active = False

        for i, node in enumerate(layers[0]):
            node.active = input[i]

        for layer in layers:
            for node in layer:
                if node.active:
                    for receiver in node.outputtingTo:
                        self.nodes[receiver].active = True



    def toDirichlet(self):
        pass

    def model(self):
        pass

    def guide(self):
        pass

    def generate(self):
        pass