import numpy as np

from NeuroEvo.Genome.NodeGene import NodeGene
from NeuroEvo.NeuralNetwork import NeuralNetwork

from NeuroEvo.NeuralNetwork.NeuralNetwork import NeuralNetwork
import torch


# Stores the architecture of a neural network
class Genome:

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.edges = []
        self.nodes = []
        self.nodeCounter = 0

        self.sendingNodes = []
        for i in range(inputSize):
            node = NodeGene(self.nodeCounter)
            self.sendingNodes.append(node)
            self.nodes.append(node)
            self.nodeCounter += 1

        self.receivingNodes = []
        for i in range(outputSize):
            node = NodeGene(self.nodeCounter)
            self.receivingNodes.append(node)
            self.nodes.append(node)
            self.nodeCounter += 1

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
        print("allocated layers")
        # Group nodes into lists belonging to their respective layer
        layerGroups = []
        for i in range(int(maxLayer)+1):
            group = []
            for i2,layer in enumerate(layerNrs):
                if(layer == i):
                    group.append(i2)
            layerGroups.append(group)

        print("naise")

        # Create matrices with weights from each layer to the layers after
        layers = []
        for fromI, layer in enumerate(layerGroups):
            layerWeights = []
            layerBiases = []
            for toI in range(fromI + 1, len(layerGroups)):
                ins = []
                outs = []
                weights = []
                for edge in self.edges:
                    if (layerNrs[edge.fromNr] == fromI and layerNrs[edge.toNr] == toI):
                        if not(layer.__contains__(edge.fromNr) and layerGroups[toI].__contains__(edge.toNr)):
                            continue
                        ins.append(layer.index(edge.fromNr))
                        outs.append(layerGroups[toI].index(edge.toNr))
                        weights.append(edge.weight)

                inOuts = torch.LongTensor([ins, outs])
                weights = torch.FloatTensor(weights)
                layerWeights.append(torch.sparse.FloatTensor(inOuts, weights, torch.Size([len(layer),
                                                                                          len(layerGroups[toI])])).to_dense().t())
                layerBiases.append(torch.tensor(np.zeros(len(layerGroups[toI]))))
            layers.append(list(zip(layerWeights, layerBiases)))

        return NeuralNetwork(layers, True)

