
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
        fromI = random.randint(0, len(self.nodes) - 1)
        while(self.nodes[fromI].output):
            fromI = random.randint(0, len(self.nodes) - 1)
        toI = random.randint(self.inputSize, len(self.nodes) - 1)

        maxTries = 1000
        while ( self.nodes[fromI].output or self.nodes[toI].input
                or self.nodes[fromI].outputtingTo.__contains__(toI)
                or (self.nodes[toI].layer <= self.nodes[fromI].layer and not self.nodes[toI].output)
                or toI == fromI):

            if(maxTries == 0):
                return
            maxTries -= 1

            while (self.nodes[fromI].output):
                fromI = random.randint(0, len(self.nodes) - 1)
            toI = random.randint(self.inputSize, len(self.nodes) - 1)

        self.nodes[fromI].outputtingTo.append(toI)
        self.edges.append(ConnectionGene.EdgeGene(fromI, toI, ((random.random()*2)-1), enabled= True, hMarker = hMarker))
        self.increaseLayers(self.nodes[fromI], self.nodes[toI])


    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self, hMarker):
        node = NodeGene.NodeGene(len(self.nodes))
        self.nodes.append(node)
        self.specifiyEdge(self.edges[random.randint(0, len(self.edges)-1)], node, hMarker)

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)

    # Add the Edge to an adding Node
    def specifiyEdge(self, edge, newNode, hMarker):
        edge.deactivate()
        print(edge.toNr)
        self.nodes[edge.fromNr].outputtingTo.remove(edge.toNr)
        self.nodes[edge.fromNr].outputtingTo.append(newNode.nodeNr)

        self.edges.append(ConnectionGene.EdgeGene(edge.fromNr, newNode.nodeNr, 1, enabled= True, hMarker = hMarker))
        self.edges.append(ConnectionGene.EdgeGene(newNode.nodeNr, edge.toNr, edge.weight, enabled= True, hMarker= (hMarker + 1)))

        newNode.outputtingTo.append(edge.toNr)
        self.increaseLayers(self.nodes[edge.fromNr], newNode)

    def increaseLayers(self,fromNode, toNode):
        if(fromNode.layer >= toNode.layer):
            toNode.layer = fromNode.layer + 1
            print("outputs: " + str(toNode.outputtingTo))
            for nodeNr in toNode.outputtingTo:
                self.increaseLayers(toNode, self.nodes[nodeNr])
            self.maxLayer = max(toNode.layer, self.maxLayer)

    def copy(self):
        g = NEATGenome(0, 0)
        g.inputSize = self.inputSize
        g.outputSize = self.outputSize
        g.maxLayer = self.maxLayer

        edges = []
        for edge in self.edges:
            edges.append(edge.copy())
        nodes = []
        for node in self.nodes:
            nodes.append(node.copy())

        g.edges = edges
        g.nodes = nodes
        g.fitness = self.fitness

        return g