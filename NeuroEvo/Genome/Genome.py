import numpy as np
from NeuralNetwork.NeuralNetwork import NeuralNetwork



# Stores the architecture of a neural network
class Genome:

    NodeGenes = np.array()
    ConnectionGene = np.array()

    def __init__(self):
        self.edges = []
        self.nodes = []
        return

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(genome):
        pass

    # Add an edge connection two nodes
    def addEdge(genome):
        pass

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(genome):
        pass

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(genome):
        pass

    # Load a Genome from the disk
    def load(self):
        return

    # Save a Genome to the disk
    def save(self):
        return

    # Returns a pytorch neural network from the genome
    def toNN(self):
        layerNrs = np.zeros(len(self.nodes))
        maxLayer = 0
        # Determine the layers to which the nodes belong, based on the assumption that a connection is always
        # towards the next layer
        notDone = False
        while(notDone):
            notDone = False
            for edge in self.edges:
                if(layerNrs[edge.fromNr] <= layerNrs[edge.toNr]):
                    notDone = True
                    layerNrs[edge.toNr] = self.nodes[edge.fromNr] + 1
                    maxLayer = max(maxLayer, layerNrs[edge.toNr])
        layers = np.array()
        for layerNr in layerNrs:
            layers.add
        for edge in self.edges:
            weights[edge.fromNr][edge.toNr] = edge.weight


        nn = NeuralNetwork(layerNrs)

        return
