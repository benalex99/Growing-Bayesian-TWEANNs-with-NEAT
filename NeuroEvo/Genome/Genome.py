import numpy as np
<<<<<<< Updated upstream
from NeuroEvo.NeuralNetwork import NeuralNetwork

=======
from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
import torch
>>>>>>> Stashed changes


# Stores the architecture of a neural network
class Genome:

    def __init__(self):
        self.edges = []
        self.nodes = []
        return

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self):
        pass

    # Add an edge connection two nodes
    def addEdge(self):
        pass

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self):
        pass

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self):
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
        notDone = True
        while(notDone):
            notDone = False
            for edge in self.edges:
                if(layerNrs[edge.fromNr] >= layerNrs[edge.toNr]):
                    notDone = True
                    layerNrs[edge.toNr] = layerNrs[edge.fromNr] + 1
                    maxLayer = max(maxLayer, layerNrs[edge.toNr])

        # Group nodes into lists belonging to their respective layer
        layerGroups = []
        for i in range(int(maxLayer)):
            group = []
            for i2,layer in enumerate(layerNrs):
                group.append(i2)
            layerGroups.append(group)

        print("naise")

        # Fill the weight matrices
        matrices = []
        biases = []
        for i,layer in enumerate(layerGroups):
            ins = []
            outs = []
            weights = []
            for edge in self.edges:
                if(layerNrs[edge.fromNr] == i):
                    ins.append(edge.fromNr)
                    outs.append(edge.toNr)
                    weights.append(edge.weight)

            inOuts = torch.LongTensor([ins,outs])
            weights = torch.FloatTensor(weights)
            matrices.append(torch.sparse.FloatTensor(inOuts, weights).to_dense())
            biases.append(torch.tensor(np.zeros(len(layer))))

        return NeuralNetwork(list(zip(matrices, biases)))

