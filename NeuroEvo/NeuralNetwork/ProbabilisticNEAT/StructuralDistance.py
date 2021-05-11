import time

import torch
from pyro.distributions import *
from tqdm import tqdm
import math

class DistanceMetric():
    @staticmethod
    def run(genome1, genome2, iterations=20000, maxTime=60, typeMatters=False):
        # Have the smaller genome be genome1
        if len(genome1.nodes) > len(genome2.nodes):
            largerGenome = genome2
            smallerGenome = genome1
        else:
            largerGenome = genome1
            smallerGenome = genome2

        # Initialize candidate population
        candidateMappings = [NodeMapping(largerGenome, smallerGenome, typeMatters=typeMatters)]
        candidateMappings[0].evaluate()

        bestScore = math.inf
        start = time.time()
        # Perform greedy search to find the optimal mapping and its score
        # Pick the mapping with the lowest distance, and create new mappings based on it
        for i in range(iterations):
            if len(candidateMappings) == 0 or bestScore == 0 or (time.time() - start) > maxTime:
                return bestScore
            candidateMappings.sort(key=lambda x: x.score, reverse=False)
            bestScore = min(bestScore, candidateMappings[0].score)
            candidateMappings.extend(candidateMappings.pop(0).children())

        return bestScore

class NodeMapping():
    def __init__(self, smallerGenome, largerGenome, allocations = None, typeMatters = False):
        self.smallerGenome = smallerGenome
        self.largerGenome = largerGenome
        if allocations == None:
            self.allocations = {}
        else:
            self.allocations = allocations
        self.typeMatters = typeMatters

        self.freeGenome1Nodes = []
        self.freeGenome2Nodes = []
        for i, node in enumerate(self.smallerGenome.nodes):
            self.freeGenome1Nodes.append(i)
        for i, node in enumerate(self.largerGenome.nodes):
            self.freeGenome2Nodes.append(i)

    # Creates new candidates based on this candidate
    def children(self):
        if len(self.allocations) == len(self.smallerGenome.nodes):
            return []
        else:
            children = []
            node = self.freeGenome1Nodes.pop()
            for mapTo in self.freeGenome2Nodes:
                # Copy the current allocations, add the new allocation and create a child
                allocation = self.allocations.copy()
                allocation[node] = mapTo
                child = NodeMapping(self.smallerGenome, self.largerGenome, allocation, typeMatters=self.typeMatters)

                # Remove node mapping possibilities from the child
                child.freeGenome1Nodes = self.freeGenome1Nodes.copy()
                freeGenome2Nodes = self.freeGenome2Nodes.copy()
                freeGenome2Nodes.remove(mapTo)
                child.freeGenome2Nodes = freeGenome2Nodes

                child.evaluate()
                children.append(child)
            return children

    # Evaluates the current candidate
    def evaluate(self):
        overlap = torch.ones(len(self.smallerGenome.nodes))
        for index, node in enumerate(self.smallerGenome.nodes):
            matchingEdges = 0
            # If node has not been mapped yet, skip checking the alignment. Assume worst case
            if not list(self.allocations.keys()).__contains__(index):
                overlap[index] = 0
                continue
            otherNode = self.largerGenome.nodes[self.allocations[index]]
            # If node types do not match, and it matters, skip checking the alignment. Assume worst case
            if self.typeMatters and node.type != otherNode.type:
                overlap[index] = 0
                continue
            for receivingNode in node.outputtingTo:
                # Check if the receiving node exists in the mapping
                if list(self.allocations.keys()).__contains__(receivingNode):
                    # Check if the connections of the node equal the mapped connections of the otherNode.
                    if otherNode.outputtingTo.__contains__(self.allocations[receivingNode]):
                        if self.typeMatters and self.smallerGenome.nodes[receivingNode].type != \
                                self.largerGenome.nodes[self.allocations[receivingNode]].type:
                            continue
                        matchingEdges += 2
            # Ratio between overlapping connections and all connections
            overlap[index] = matchingEdges / max((len(node.outputtingTo) + len(otherNode.outputtingTo)), 1)
            # If both have no outgoing edges, they are equal
            if (len(node.outputtingTo) + len(otherNode.outputtingTo)) == 0:
                overlap[index] = 1

        # Difference between possible alignment and actual alignment
        self.score = len(self.largerGenome.nodes)-sum(overlap).item()
        return self.score


