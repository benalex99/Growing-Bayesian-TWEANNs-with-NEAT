
# Decorator for the Genome class. Wraps around the Genome to give it a score, so it can be used in NEAT.
# Needs to implement a Genome base class.
import math

import numpy as np
import torch

from NeuroEvo.Genome import Genome, NodeGene, ConnectionGene
import random
import copy
from pyro.distributions import *


class NEATGenome(Genome.Genome):

    def __init__(self, inputSize, outputSize):
        super(NEATGenome, self).__init__(inputSize, outputSize)
        self.fitness = -math.inf
        self.adjustedFitness = -math.inf

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self, hMarker):
        if (len(self.edges) == 0):
            self.addEdge(hMarker)
            return 1

        randomMutate = Categorical(torch.tensor([0.4, 0.3, 0.3])).sample([1])[0]
        if randomMutate == 0:
            self.addEdge(hMarker)
            return 1
        elif randomMutate == 1:
            self.addNode(hMarker)
            return 3
        else:
            self.tweakWeight(1)
            return 0

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
        self.edges.append(ConnectionGene.EdgeGene(fromI, toI, ((random.random()*2)-1), enabled= True, hMarker = hMarker))
        self.increaseLayers(self.nodes[fromI], self.nodes[toI])

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self, hMarker):
        edge = self.edges[random.randint(0, len(self.edges)-1)]

        while(not edge.enabled):
            edge = self.edges[random.randint(0, len(self.edges) - 1)]

        node = NodeGene.NodeGene(nodeNr=len(self.nodes))
        self.nodes.append(node)
        self.specifiyEdge(edge, node, hMarker)

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)

    # Add the incoming and outgoing edges to the newly added intervening Node
    def specifiyEdge(self, edge, newNode, hMarker):
        edge.deactivate()
        self.nodes[edge.fromNr].outputtingTo.remove(edge.toNr)
        self.nodes[edge.fromNr].outputtingTo.append(newNode.nodeNr)

        self.edges.append(ConnectionGene.EdgeGene(edge.fromNr, newNode.nodeNr, 1, enabled= True, hMarker = hMarker))
        self.edges.append(ConnectionGene.EdgeGene(newNode.nodeNr, edge.toNr, edge.weight, enabled= True, hMarker= (hMarker + 1)))

        newNode.outputtingTo.append(edge.toNr)
        self.increaseLayers(self.nodes[edge.fromNr], newNode)

    def increaseLayers(self,fromNode, toNode):
        if(fromNode.layer >= toNode.layer):
            toNode.layer = fromNode.layer + 1
            for nodeNr in toNode.outputtingTo:
                self.increaseLayers(toNode, self.nodes[nodeNr])
            self.maxLayer = max(toNode.layer, self.maxLayer)

    def copy(self):
        g = NEATGenome(0, 0)
        g.inputSize = self.inputSize
        g.outputSize = self.outputSize
        g.maxLayer = self.maxLayer

        g.edges = copy.deepcopy(self.edges)
        g.nodes = copy.deepcopy(self.nodes)
        g.fitness = self.fitness
        return g