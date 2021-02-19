
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
            return self.addEdge(hMarker)

        randomMutate = random.randint(0, 2)
        if randomMutate == 0:
            return self.addEdge(hMarker)
        elif randomMutate == 1:
            return self.addNode(hMarker)
        else:
            return self.tweakWeight(0.1)


    # Add an edge to connect two nodes
    def addEdge(self, hMarker):
        fromI = random.randint(0, len(self.sendingNodes)-1)
        toI = random.randint(0, len(self.receivingNodes)-1)

        while ((not (self.inputSize <= self.receivingNodes[toI].nodeNr < self.inputSize + self.outputSize))
               and self.receivingNodes[toI].nodeNr <= self.sendingNodes[fromI].nodeNr):
            fromI = random.randint(0, len(self.sendingNodes) - 1)
            toI = random.randint(0, len(self.receivingNodes)-1)

        self.edges.append(ConnectionGene.EdgeGene(self.sendingNodes[fromI].nodeNr,
                                                  self.receivingNodes[toI].nodeNr, ((random.random()*2)-1)), hMarker)
        return 1

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self, hMarker):
        node = NodeGene.NodeGene(self.nodeCounter)
        self.nodeCounter += 1
        self.nodes.append(node)
        self.sendingNodes.append(node)
        self.receivingNodes.append(node)
        self.specifiyEdge(self.edges[random.randint(0, len(self.edges)-1)], node, hMarker)
        return 2

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)
        return 0

    # Add the Edge to an adding Node
    def specifiyEdge(self, edgeToSpecifiy, nodeToAppend, hMarker):
        self.edges.append(ConnectionGene.EdgeGene(edgeToSpecifiy.fromNr, nodeToAppend.nodeNr, 1), hMarker)
        self.edges.append(ConnectionGene.EdgeGene(nodeToAppend.nodeNr, edgeToSpecifiy.toNr, edgeToSpecifiy.weight), (hMarker+1))

        edgeToSpecifiy.deactivate()
