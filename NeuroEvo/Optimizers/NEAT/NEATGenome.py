
# Decorator for the Genome class. Wraps around the Genome to give it a score, so it can be used in NEAT.
# Needs to implement a Genome base class.
import numpy as np

from Genome import Genome, NodeGene, ConnectionGene
import random


class NEATGenome(Genome.Genome):

    def __init__(self):
        super(NEATGenome, self).__init__()

    # Returns a random Index Number to get.
    def randomIndex(self, nodeArray) -> float:
        return nodeArray[random.randint(0, len(nodeArray)-1)].nodeNr

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self):
        randomMutate = random.randint(0, 2)
        if randomMutate == 0:
            self.addEdge()
        elif randomMutate == 1:
            self.addNode()
        else:
            self.tweakWeight(0.1)

    # Add an edge to connect two nodes
    def addEdge(self):
        firstNodeToConnect = self.randomIndex(self.nodes)
        secondNodeToConnect = self.randomIndex(self.nodes)
        while secondNodeToConnect == firstNodeToConnect:
            secondNodeToConnect = self.randomIndex(self.nodes)
        self.edges.append(ConnectionGene.EdgeGene(firstNodeToConnect, secondNodeToConnect, ((random.random()*2)-1)))

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self):
        self.nodes.append(NodeGene.NodeGene(len(self.nodes)))
        return

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)
