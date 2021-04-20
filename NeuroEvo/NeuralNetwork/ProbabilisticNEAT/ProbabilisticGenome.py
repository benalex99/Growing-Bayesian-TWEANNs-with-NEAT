import copy
import math
import random

from NeuroEvo.Genome.Visualizer import Visualizer
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
        self.nodes = []
        self.maxLayer = 1

        for i in range(inputSize):
            node = AdvancedNodeGene.AdvancedNodeGene(len(self.nodes),layer = 0, input = True)
            self.nodes.append(node)

        for i in range(outputSize):
            node = AdvancedNodeGene.AdvancedNodeGene(len(self.nodes),layer = 1, output = True)
            self.nodes.append(node)

        self.fitness = -math.inf

    # TODO: Add "change output node type"
    # TODO: Last layer nodes MUST be of distribution type
    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self, hMarker):
        # If we have an empty network, the only option is to add an edge for throughput
        if (len(self.edges) == 0):
            self.addEdge(hMarker)
            return 1

        # Choose the type of mutation and execute it
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

        newEdge = AdvancedEdgeGene.AdvancedEdgeGene(fromI, toI, weight = 1, hMarker = hMarker)

        # Attempt to find a valid new edge for maxTries times
        invalid = True
        maxTries = 1000
        while not self.validEdge(newEdge) or invalid:
            if(maxTries == 0):
                return
            maxTries -= 1

            # Quickly iterate for semi valid input and output nodes
            while (self.nodes[fromI].output):
                fromI = random.randint(0, len(self.nodes) - 1)
            while (self.nodes[toI].input and outNode == None):
                toI = random.randint(self.inputSize, len(self.nodes) - 1)

            # Create a new edge with the given sending and receiving nodes
            newEdge = AdvancedEdgeGene.AdvancedEdgeGene(fromI,toI,weight=((random.random()*2)-1), hMarker = hMarker)

            # If sending node is a dirichlet, specify which class output of the dirichlet is sent
            if self.nodes[fromI].type == NodeType.Dirichlet:
                newEdge.fromClass = random.randint(0, self.nodes[fromI].classCount - 1)

            # If receiving nodes are dirichlet or categorical, specify the target classes
            if self.nodes[toI].type == NodeType.Dirichlet or self.nodes[toI].type == NodeType.Categorical:
                newEdge.toClass = self.nodes[toI].classCount

            # If the receiving node is a Gaussian that already has two connections, declare the connection invalid
            if self.nodes[toI].type == NodeType.Gaussian:
                if self.nodes[toI].classCount == 2:
                    invalid = True
                else:
                    # If the gaussian does not yet have a variance input (default is 1), then declare the connection valid
                    # and assign the variance index
                    newEdge.toClass = 1
                    invalid = False
            else:
                invalid = False

        # If the final receiving node is of type dirichlet, categorical or gaussian
        if (self.nodes[toI].type == NodeType.Dirichlet
                or self.nodes[toI].type == NodeType.Categorical
                or self.nodes[toI].type == NodeType.Gaussian):
            # Add increase the class counter
            self.nodes[toI].classCount += 1

        # Finish adding the edge
        self.nodes[fromI].outputtingTo.append(toI)
        self.edges.append(newEdge)
        self.increaseLayers(self.nodes[fromI], self.nodes[toI])

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(self, hMarker):
        # Search for an active edge to replace
        edge = self.edges[random.randint(0, len(self.edges)-1)]
        while(not edge.enabled):
            edge = self.edges[random.randint(0, len(self.edges) - 1)]

        # Add a node with random node type
        node = AdvancedNodeGene.AdvancedNodeGene(nodeNr=len(self.nodes), type=NodeType.random())
        self.nodes.append(node)

        # Manage the connection changes
        self.specifiyEdge(edge, node, hMarker)

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, weight):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, weight)

    # Add the incoming and outgoing edges to the newly added intervening Node
    def specifiyEdge(self, edge, newNode, hMarker):
        # Bookkeeping
        edge.deactivate()
        self.nodes[edge.fromNr].outputtingTo.remove(edge.toNr)
        self.nodes[edge.fromNr].outputtingTo.append(newNode.nodeNr)

        # Create an edge from the original sending node and class to the new node
        firstEdge = AdvancedEdgeGene.AdvancedEdgeGene(edge.fromNr, newNode.nodeNr, weight=1, hMarker = hMarker, fromClass=edge.fromClass)
        self.edges.append(firstEdge)

        # Create an edge from the new node to the original receiving node and class
        secondEdge = AdvancedEdgeGene.AdvancedEdgeGene(newNode.nodeNr, edge.toNr, edge.weight, hMarker= (hMarker + 1), toClass=edge.toClass)
        self.edges.append(secondEdge)

        # If the new node is a dirichlet specify the dirichlets parameter and output class indices for both edges
        if newNode.type == NodeType.Dirichlet:
            firstEdge.toClass = 0
            secondEdge.fromClass = 0

        # If new node is a categorical or gaussian specify the output class index for the new first new edge
        # For a categorical this is simply the first classes parameter.
        # For a gaussian this is the mean parameter.
        if newNode.type == NodeType.Categorical or newNode.type == NodeType.Gaussian:
            firstEdge.toClass = 0

        if newNode.type == NodeType.Gaussian:
            firstEdge.toClass = 0

        # Bookkeeping
        newNode.outputtingTo.append(edge.toNr)
        self.increaseLayers(self.nodes[edge.fromNr], newNode)

    # Implements the probabilistic forward model for the SVI optimizer
    def model(self, data):
        # Get the nodes layer allocation for update sequencing
        layers = self.getLayers()

        # Splitting data into features and predictions
        inputData = data[:self.inputSize, :]
        outputData = data[self.inputSize:self.inputSize+self.outputSize, :]

        for input, output in zip(inputData, outputData):

            # Loop through all but the last layer, as this one will be uniquely sampled with our data
            for i, layer in enumerate(layers):
                # If we are in the first layer, append our feature data to the inputs of the input nodes
                if(i == 0):
                    for i2, nodeNr in enumerate(layer):
                        self.nodes[nodeNr].inputs.append(input[i2])

                # If we are in the last layer, sample our output nodes predictions with our observations
                if (i == len(layers) - 1):
                    outputs = []
                    for i, (outputPoint, nodeNr) in enumerate(zip(output, layer)):
                        pyro.sample("obs_{}".format(i), self.nodes[nodeNr].function(), obs=outputPoint)

                # For each node in the layer calculate the output and distribute it to the other nodes
                for nodeNr in layer:
                    # Calculate the output
                    if not self.nodes[nodeNr].input and self.nodes[nodeNr].type != NodeType.Relu and self.nodes[nodeNr].type != NodeType.Multiplication:
                        output = pyro.sample("node" + str(nodeNr), self.nodes[nodeNr].function())
                    else:
                        output = self.nodes[nodeNr].function()

                    # Distribute weighted output to other nodes
                    for nodeNr2 in list(dict.fromkeys(self.nodes[nodeNr].outputtingTo)):
                        for edge in self.edges:
                            if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                if self.nodes[nodeNr].type != NodeType.Dirichlet:
                                    self.nodes[nodeNr2].inputs.append([output * edge.weight, edge.toClass])
                                else:
                                    self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])

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

    # Checks if the proposed edge is valid
    def validEdge(self, edge):

        # Output may not send back to other nodes
        if (self.nodes[edge.fromNr].output
                # Input nodes may not receive back
                or self.nodes[edge.toNr].input
                # Edge may not already exist
                or self.edgeExists(edge)
                or (self.nodes[edge.toNr].layer <= self.nodes[edge.fromNr].layer)
                or edge.toNr == edge.fromNr
                or self.nodes[edge.fromNr].type == NodeType.Dirichlet):
            return False
        return True

    # Checks if the proposed edge already exists
    def edgeExists(self, edge):
        for oldEdge in self.edges:
            if (oldEdge.fromNr == edge.fromNr
                    and oldEdge.toNr == edge.toNr
                    and oldEdge.fromClass == edge.fromClass
                    and oldEdge.toClass == edge.toClass):
                return True
        return False

    # Implements the probabilistic forward model for the SVI optimizer
    def generate(self, inputData):
        # Get the nodes layer allocation for update sequencing
        layers = self.getLayers()

        # Splitting data into features and predictions
        outputs = []
        for input in inputData:
            # Loop through all but the last layer, as this one will be uniquely sampled with our data
            for i, layer in enumerate(layers):
                # If we are in the first layer, append our feature data to the inputs of the input nodes
                if(i == 0):
                    for i2, nodeNr in enumerate(layer):
                        self.nodes[nodeNr].inputs.append(input[i2])

                # If we are in the last layer, sample our output nodes predictions with our observations
                if (i == len(layers) - 1):
                    output = []
                    for nodeNr in layer:
                        if not self.nodes[nodeNr].input and self.nodes[nodeNr].type != NodeType.Relu and self.nodes[nodeNr].type != NodeType.Multiplication:
                            y = self.nodes[nodeNr].function().sample([1])[0].tolist()
                        else:
                            y = self.nodes[nodeNr].function()
                        output.append(y)
                    outputs.append(output)

                # For each node in the layer calculate the output and distribute it to the other nodes
                for nodeNr in layer:
                    #print("NodeNr: " + str(nodeNr))
                    # Calculate the output
                    if self.nodes[nodeNr].type != NodeType.Relu and self.nodes[nodeNr].type != NodeType.Multiplication:
                        output = self.nodes[nodeNr].function().sample([1]).tolist()[0]
                    else:
                        output = self.nodes[nodeNr].function()
                    #print("Output: " + str(output))
                    # Distribute weighted output to other nodes
                    for nodeNr2 in self.nodes[nodeNr].outputtingTo:
                        # Dirichlet distribution has a different output format and must be handled differently
                        if self.nodes[nodeNr].type != NodeType.Dirichlet:
                            for edge in self.edges:
                                if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                    self.nodes[nodeNr2].inputs.append([output * edge.weight, edge.toClass])
                        else:
                            for edge in self.edges:
                                if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                    if self.nodes[nodeNr].classCount > 1:
                                        self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])
                                    else:
                                        self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])
        return outputs

    def nodeStats(self):
        dirichlets = 0
        categoricals = 0
        gaussians = 0
        relus = 0
        multiplications = 0
        for node in self.nodes:
            if node.type == NodeType.Dirichlet:
                dirichlets += 1
            if node.type == NodeType.Categorical:
                categoricals += 1
            if node.type == NodeType.Gaussian:
                gaussians += 1
            if node.type == NodeType.Relu:
                relus += 1
            if node.type == NodeType.Multiplication:
                multiplications += 1

        return "Dirichlets: " + str(dirichlets) + "\n" + \
                "Categoricals: " + str(categoricals) + "\n" + \
                "Gaussians: " + str(gaussians) + "\n" + \
                "Relus: " + str(relus) + "\n" + \
                "Multiplications: " + str(multiplications)

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

        labels = {}
        for node, nodePos in zip(nodes, nodePositions):
            G.addNode(node, pos = nodePos)
            if(self.nodes[node].input):
                labels[node] = "inp"
            else:
                if(self.nodes[node].output):
                    labels[node] = "outp"
                else:
                    labels[node] = str(self.nodes[node].type)

        G.visualize(ion= ion, labels=labels)