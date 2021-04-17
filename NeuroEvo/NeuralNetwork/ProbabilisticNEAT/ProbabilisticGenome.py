import copy
import math
import random

from NeuroEvo.Genome import Genome
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT import AdvancedNodeGene
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT import AdvancedEdgeGene
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.AdvancedNodeGene import NodeType
import torch
import pyro
import numpy as np


class ProbabilisticGenome(Genome.Genome):
    def __init__(self, inputSize, outputSize):
        super(ProbabilisticGenome, self).__init__(inputSize, outputSize)
        self.fitness = -math.inf

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self, hMarker):
        if (len(self.edges) == 0):
            self.addEdge(hMarker)
            return 1

        randomMutate = random.randint(0, 2)
        if randomMutate == 0:
            self.addEdge(hMarker)
            return 1
        elif randomMutate == 1:
            self.addNode(hMarker)
            return 3
        else:
            self.tweakWeight(1)
            return 0

    # TODO: Change edge adding for dirichlet, categorical and gaussian nodes
    # Add an edge to connect two nodes
    def addEdge(self, hMarker, outNode = None):
        fromI = random.randint(0, len(self.nodes) - 1)
        while(self.nodes[fromI].output):
            fromI = random.randint(0, len(self.nodes) - 1)

        if(outNode == None):
            toI = random.randint(self.inputSize, len(self.nodes) - 1)
            while(self.nodes[toI].input):
                toI = random.randint(self.inputSize, len(self.nodes) - 1)
        else:
            toI = outNode.nodeNr

        maxTries = 1000
        while ( self.nodes[fromI].output
                or self.nodes[toI].input
                or self.nodes[fromI].outputtingTo.__contains__(toI)
                or (self.nodes[toI].layer <= self.nodes[fromI].layer and not self.nodes[toI].output)
                or toI == fromI):

            if(maxTries == 0):
                return
            maxTries -= 1

            while (self.nodes[fromI].output):
                fromI = random.randint(0, len(self.nodes) - 1)
            while (self.nodes[toI].input and outNode == None):
                toI = random.randint(self.inputSize, len(self.nodes) - 1)

        self.nodes[fromI].outputtingTo.append(toI)
        self.edges.append(AdvancedEdgeGene.AdvancedEdgeGene(fromI, toI, ((random.random()*2)-1), enabled= True, hMarker = hMarker))
        self.increaseLayers(self.nodes[fromI], self.nodes[toI])

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    # TODO: Last layer nodes MUST be of distribution type
    def addNode(self, hMarker):
        # Search for an active edge to replace
        edge = self.edges[random.randint(0, len(self.edges)-1)]
        while(not edge.enabled):
            edge = self.edges[random.randint(0, len(self.edges) - 1)]

        # Initialize a node with random node type
        node = AdvancedNodeGene.AdvancedNodeGene(nodeNr=len(self.nodes), type=NodeType.random())
        self.nodes.append(node)

        # Manage the connection changes
        self.specifiyEdge(edge, node, hMarker)

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)

    # TODO: Change edge adding dirichlet, categorical and gaussian nodes
    # Add the incoming and outgoing edges to the newly added intervening Node
    def specifiyEdge(self, edge, newNode, hMarker):
        edge.deactivate()
        self.nodes[edge.fromNr].outputtingTo.remove(edge.toNr)
        self.nodes[edge.fromNr].outputtingTo.append(newNode.nodeNr)

        self.edges.append(AdvancedEdgeGene.AdvancedEdgeGene(edge.fromNr, newNode.nodeNr, 1, enabled= True, hMarker = hMarker))
        self.edges.append(AdvancedEdgeGene.AdvancedEdgeGene(newNode.nodeNr, edge.toNr, edge.weight, enabled= True, hMarker= (hMarker + 1)))

        newNode.outputtingTo.append(edge.toNr)
        self.increaseLayers(self.nodes[edge.fromNr], newNode)

    # Implements the probabilistic forward model for the SVI optimizer
    def model(self, data):
        # Get the nodes layer allocation for update sequencing
        layers = self.getLayers()

        # Loop through all but the last layer, as this one will be uniquely sampled with our data
        for i, layer in enumerate(layers):
            # Skip layer if it is the last
            if (i == len(layers) - 1):
                break
            # For each node in the layer calculate the output and distribute it to the other nodes
            for nodeNr in layer:
                # Calculate the output
                if self.nodes[nodeNr].type != NodeType.Relu and self.nodes[nodeNr].type != NodeType.Multiplication:
                    output = pyro.sample(str(nodeNr), self.nodes[nodeNr].function())
                else:
                    output = self.nodes[nodeNr].function()

                # Distribute weighted output to other nodes
                for nodeNr2 in self.nodes[nodeNr].outputtingTo:
                    for edge in self.edges:
                        if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                            if self.nodes[nodeNr].type != NodeType.Dirichlet:
                                self.nodes[nodeNr2].inputs.append([output * edge.weight, edge.toClass])
                            else:
                                self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])

        # Sample the last layers nodes with our data
        for nodeNr in layers[len(layers) - 1]:
            pyro.sample("obs", self.nodes[nodeNr].function(), obs=data)

    # TODO: implement :)
    def guide(self):
        pass

    def copy(self):
        g = ProbabilisticGenome(0, 0)
        g.inputSize = self.inputSize
        g.outputSize = self.outputSize
        g.maxLayer = self.maxLayer

        g.edges = copy.deepcopy(self.edges)
        g.nodes = copy.deepcopy(self.nodes)
        g.fitness = self.fitness
        return g

    # Update the nodes layerings
    def increaseLayers(self, fromNode, toNode):
        if(fromNode.layer >= toNode.layer):
            toNode.layer = fromNode.layer + 1
            for nodeNr in toNode.outputtingTo:
                self.increaseLayers(toNode, self.nodes[nodeNr])
            self.maxLayer = max(toNode.layer, self.maxLayer)