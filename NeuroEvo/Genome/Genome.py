import numpy as np

from NeuroEvo.Genome.Visualizer import Visualizer
from NeuroEvo.Genome.NodeGene import NodeGene
from NeuroEvo.Genome.ConnectionGene import EdgeGene
from NeuroEvo.Genome.NeuralNetwork import NeuralNetwork

import torch
import copy

# Stores the architecture of a neural network
class Genome():

    def __init__(self, inputSize, outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.edges = []
        self.nodes = []
        self.maxLayer = 1

        for i in range(inputSize):
            node = NodeGene(len(self.nodes),layer = 0, input = True)
            self.nodes.append(node)

        for i in range(outputSize):
            node = NodeGene(len(self.nodes),layer = 1, output = True)
            self.nodes.append(node)

    # Add another input node to increase the input dimensionality
    def increaseInput(self):
        self.inputSize += 1
        node = NodeGene(len(self.nodes),layer = 0, input = True)
        self.nodes.append(node)

    # Add another outputNode to increase the output dimensionality
    def increaseOutput(self):
        self.outputSize += 1
        node = NodeGene(len(self.nodes),layer = 1, output = True)
        self.nodes.append(node)

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self, hMarker):
        pass

    # Add an edge connection two nodes
    def addEdge(self, hMarker):
        pass

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self, hMarker):
        pass

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
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
        self.edgeLocations = []
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
                layerWeights.append(torch.sparse.FloatTensor(inOuts, weights,
                                                        torch.Size([len(layer), len(layerGroups[toI])])
                                                             ).to_dense().t())
                layerBiases.append(torch.tensor(np.zeros(len(layerGroups[toI]))))

            layers.append(list(zip(layerWeights, layerBiases)))

        return NeuralNetwork(layers, True)

    # Takes weights from a pytorch nn and updates our genomes edge weights from it
    def weightsFromNN(self, nn):
        layers = nn.fromToLayers
        layerGroups = self.getLayers()

        # TODO: Can we make this less terribly cascaded loops?
        newWeights = []
        # Add connections
        for layerNr1, fromLayer in enumerate(layers):
           for layerNr2, toLayer in enumerate(fromLayer):
               for index1, outputVector in enumerate(toLayer.weight):
                   for index2, weight in enumerate(outputVector.data):
                       if weight != 0:
                           newWeights.append([layerGroups[layerNr1][index2],
                                           layerGroups[layerNr2 + layerNr1 + 1][index1],
                                                       weight.item()])
        # TODO: Optimize finding the edges by keeping them indexed based on their input and output
        # TODO: For O(1) access performance instead of O(edges/2)
        # Assign weights to old edges
        for edge in self.edges:
            for newWeight in newWeights:
                if edge.fromNr == newWeight[0] and edge.toNr == newWeight[1]:
                    edge.weight = newWeight[2]
                    break

    # TODO: Implement updating the weights with gradients. Zero the gradients of 0 weights and frozen weights!
    # Tunes the weights using gradient descent
    def tuneWeights(self, xData, yData):
        model = self.toNN()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for t in range(iter):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(xData)
            loss = criterion(y_pred, yData)

            if t % 100 == 99:
                print("Iteration: " + str(t) + " Loss: " + str(loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Visualize the genomes graph representation
    def visualize(self, ion=True):
        groups = self.getLayers()
        G = Visualizer()
        nodePositions = []
        parents = []
        nodes = []
        for y,layer in enumerate(groups):
            for x, node in enumerate(layer):
                nodes.append(node)
                parents.append([])

        for edge in self.edges:
            if(edge.enabled):
                G.addEdge(edge.fromNr, edge.toNr)
                parents[nodes.index(edge.toNr)].append(edge.fromNr)

        for y,layer in enumerate(groups):
            if(y == 0):
                for x, node in enumerate(layer):
                    nodePositions.append((y, -len(layer)/2 + x))
            else:
                positions = []
                for x, node in enumerate(layer):
                    x = 0
                    for parent in parents[nodes.index(node)]:
                        x += nodePositions[nodes.index(parent)][1]
                    positions.append(x/max(1, len(parents[nodes.index(node)])))
                    # nodePositions.append((y,x/len(parents[node])))
                order = np.argsort(positions)
                for index in order:
                     nodePositions.append((y, -len(layer)/2 + index))

        for node, nodePos in zip(nodes, nodePositions):
            G.addNode(node, pos = nodePos)

        G.visualize(ion= ion)

    # Update the layer assignments of the nodes and return them in a list of lists
    def getLayers(self):
        for node in self.nodes:
            if not node.input:
                node.layer = max(node.layer, 1)

        for node in self.nodes:
            if node.output:
                node.layer = self.maxLayer

        # Group nodes into lists belonging to their respective layer
        layerGroups = []
        for i in range(int(self.maxLayer) + 1):
            group = []
            for i2, node in enumerate(self.nodes):
                if (node.layer == i):
                    group.append(node.nodeNr)
            # group.sort(reverse= True)
            layerGroups.append(group)
        return layerGroups

    def copy(self):
        pass

    def updateLayers(self):
        for edge in self.edges:
            self.increaseLayers(self.nodes[edge.fromNr],self.nodes[edge.toNr])

        maxLayer = 0
        for node in self.nodes:
            maxLayer = max(maxLayer, node.layer)

        for node in self.nodes:
            if node.output:
                node.layer = maxLayer

    def increaseLayers(self,fromNode, toNode):
        if(fromNode.layer >= toNode.layer):
            toNode.layer = fromNode.layer + 1
            for nodeNr in toNode.outputtingTo:
                self.increaseLayers(toNode, self.nodes[nodeNr])
            self.maxLayer = max(toNode.layer, self.maxLayer)

    def __repr__(self):
        str = "nodes: \n"
        for node in self.nodes:
            str += node.__repr__() + "\n"
        str += "edges: \n"
        for edge in self.edges:
            str += edge.__repr__() + "\n"
        return str

    def fromString(self, string):
        pass
