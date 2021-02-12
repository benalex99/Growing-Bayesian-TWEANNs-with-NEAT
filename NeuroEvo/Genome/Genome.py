import numpy as np
import torch



# Stores the architecture of a neural network
class Genome:

    NodeGenes = np.array()
    ConnectionGene = np.array()

    def __init__(self):
        return

    # Load a Genome from the disk
    def load(self):
        return

    # Save a Genome to the disk
    def save(self):
        return

    # Returns a pytorch neural network from the genome
    def toNN(self):
        return