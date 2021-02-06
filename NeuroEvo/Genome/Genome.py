import numpy as np
import torch



# Stores the architecture of a neural network
class Genome:

    NodeGenes = np.array()
    ConnectionGene = np.array()

    def __init__(self):
        return

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self):
        return

    # Add an edge connection two nodes
    def addEdge(self):
        return

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self):
        return

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self):
        return

    # Load a Genome from the disk
    def load(self):
        return

    # Save a Genome to the disk
    def save(self):
        return