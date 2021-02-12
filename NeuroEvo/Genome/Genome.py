import numpy as np
import torch



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
        return