import numpy as np

from NeuroEvo.Genome.Visualizer import Visualizer
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
        self.inputNodes = []
        for i in range(inputSize):
            node = NodeGene(self.nodeCounter)
            self.sendingNodes.append(node)
            self.nodes.append(node)
            self.inputNodes.append(node)
            self.nodeCounter += 1

        self.receivingNodes = []
        self.outputNodes = []
        for i in range(outputSize):
            node = NodeGene(self.nodeCounter)
            self.receivingNodes.append(node)
            self.nodes.append(node)
            self.outputNodes.append(node)
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
        layerGroups = self.getLayers()

        #print("naise")

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
                    if(edge.enabled):
                        if (self.nodes[edge.fromNr].layer == fromI and self.nodes[edge.toNr].layer == toI):
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

    def visualize(self):
        groups = self.getLayers()
        G = Visualizer()
        for y,layer in enumerate(groups):
            for x, node in enumerate(layer):
                G.addNode(node, pos = (y, -len(layer)/2 + x))

        for edge in self.edges:
            if(edge.enabled):
                G.addEdge(edge.fromNr, edge.toNr)
        G.visualize()

    def getLayers(self):
        self.maxLayer = 0
        # Determine the layers to which the nodes belong, based on the assumption that a connection is always
        # towards a later layer
        notDone = True
        while (notDone):
            notDone = False
            for edge in self.edges:
                if (edge.enabled):
                    if (self.nodes[edge.fromNr].layer >= self.nodes[edge.toNr].layer):
                        notDone = True
                        self.nodes[edge.toNr].layer = self.nodes[edge.fromNr].layer + 1
                        self.maxLayer = max(self.maxLayer, self.nodes[edge.toNr].layer)

        for node in self.outputNodes:
            node.layer = self.maxLayer

        # Group nodes into lists belonging to their respective layer
        layerGroups = []
        for i in range(int(self.maxLayer) + 1):
            group = []
            for i2, node in enumerate(self.nodes):
                if (node.layer == i):
                    group.append(i2)
            layerGroups.append(group)
        return layerGroups

    def copy(self):
        g = Genome(0,0)
        g.inputSize = self.inputSize
        g.outputSize = self.outputSize
        g.nodeCounter = self.nodeCounter

        sendingNodes = []
        inputNodes = []
        receivingNodes = []
        outputNodes = []
        for node in self.sendingNodes:
            sendingNodes.append(node.copy())
        for node in self.inputNodes:
            inputNodes.append(node.copy())
        for node in self.receivingNodes:
            receivingNodes.append(node.copy())
        for node in self.outputNodes:
            outputNodes.append(node.copy())

        edges = []
        for edge in self.edges:
            edges.append(edge.copy())
        nodes = []
        for node in self.nodes:
            nodes.append(node.copy())

        g.sendingNodes = sendingNodes
        g.inputNodes = inputNodes
        g.receivingNodes = receivingNodes
        g.outputNodes = outputNodes
        g.edges = edges

        return g
