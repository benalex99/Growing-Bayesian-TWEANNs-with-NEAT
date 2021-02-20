
# Decorator for the Genome class. Wraps around the Genome to give it a score, so it can be used in NEAT.
# Needs to implement a Genome base class.
import numpy as np

from NeuroEvo.Genome import Genome, NodeGene, ConnectionGene
import random


class NEATGenome(Genome.Genome):

    def __init__(self, inputSize, outputSize):
        super(NEATGenome, self).__init__(inputSize, outputSize)
        self.fitness = 0

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
            return 2
        else:
            self.tweakWeight(0.1)
            return 0



    # Add an edge to connect two nodes
    def addEdge(self, hMarker):
        fromI = random.randint(0, len(self.sendingNodes)-1)
        toI = random.randint(0, len(self.receivingNodes)-1)

        while ((not (self.inputSize <= self.receivingNodes[toI].nodeNr < self.inputSize + self.outputSize))
               and self.receivingNodes[toI].nodeNr <= self.sendingNodes[fromI].nodeNr):
            fromI = random.randint(0, len(self.sendingNodes) - 1)
            toI = random.randint(0, len(self.receivingNodes)-1)

        self.edges.append(ConnectionGene.EdgeGene(self.sendingNodes[fromI].nodeNr,
                                                  self.receivingNodes[toI].nodeNr, ((random.random()*2)-1), hMarker = hMarker))

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self, hMarker):
        node = NodeGene.NodeGene(self.nodeCounter)
        self.nodeCounter += 1
        self.nodes.append(node)
        self.sendingNodes.append(node)
        self.receivingNodes.append(node)
        self.specifiyEdge(self.edges[random.randint(0, len(self.edges)-1)], node, hMarker)

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)

    # Add the Edge to an adding Node
    def specifiyEdge(self, edgeToSpecifiy, nodeToAppend, hMarker):
        self.edges.append(ConnectionGene.EdgeGene(edgeToSpecifiy.fromNr, nodeToAppend.nodeNr, 1, hMarker = hMarker))
        self.edges.append(ConnectionGene.EdgeGene(nodeToAppend.nodeNr, edgeToSpecifiy.toNr, edgeToSpecifiy.weight, hMarker = (hMarker+1)))

        edgeToSpecifiy.deactivate()

    def copy(self):
        g = NEATGenome(0, 0)
        g.inputSize = self.inputSize
        g.outputSize = self.outputSize
        g.nodeCounter = self.nodeCounter

        sendingNodes = []
        receivingNodes = []
        for node in self.sendingNodes:
            sendingNodes.append(node.copy())
        for node in self.receivingNodes:
            receivingNodes.append(node.copy())

        edges = []
        for edge in self.edges:
            edges.append(edge.copy())
        nodes = []
        for node in self.nodes:
            nodes.append(node.copy())

        g.sendingNodes = sendingNodes
        g.receivingNodes = receivingNodes
        g.edges = edges
        g.nodes = nodes
        g.fitness = self.fitness

        return g