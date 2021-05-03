import copy
import math
import random

from pyro.infer import SVI, Trace_ELBO
from tqdm import tqdm
from pyro.distributions import *
from pyro.optim import Adam

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
            node.type = NodeType.Sum
            node.classCount = 0
            self.nodes.append(node)

        self.fitness = -math.inf

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(self, hMarker, weightsOnly=False):
        if weightsOnly:
            self.tweakWeight(1)
            return 1

        # If we have an empty network, the only option is to add an edge
        if (len(self.edges) == 0):
            self.addEdge(hMarker)
            return 1

        # Choose the type of mutation and execute it
        randomMutate = Categorical(torch.Tensor([0.6,0.3,0.1])).sample([1])[0]
        if randomMutate == 0:
            return self.addEdge(hMarker)
        elif randomMutate == 1:
            self.addNode(hMarker)
            return 3
        elif randomMutate == 2:
            self.tweakWeight()
            return 1

    # Add an edge to connect two nodes.
    # For distribution nodes, if the edge targets them, they create a new class
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
        maxTries = 5000
        invalids1 = 0
        invalids2 = 0
        while not self.validEdge(newEdge) or invalid:
            # if not self.validEdge(newEdge):
            #     invalids1 += 1
            # if invalid:
            #     invalids2 += 1
            if(maxTries == 0):
                # print("Invalid1: " + str(invalids1))
                # print("Invalid2: " + str(invalids2))
                return 0
            maxTries -= 1

            # Quickly iterate for semi valid input and output nodes
            while (self.nodes[fromI].output):
                fromI = random.randint(0, len(self.nodes) - 1)
            while (self.nodes[toI].input and outNode == None):
                toI = random.randint(self.inputSize, len(self.nodes) - 1)

            # Create a new edge with the given sending and receiving nodes
            newEdge = AdvancedEdgeGene.AdvancedEdgeGene(fromI,toI,weight=((random.random()*2)-1), hMarker = hMarker)

            # If sending node is dirichlet or categorical, specify which class output is sent
            if self.nodes[fromI].type == NodeType.Dirichlet or self.nodes[fromI].type == NodeType.Categorical:
                newEdge.fromClass = random.randint(0, self.nodes[fromI].classCount - 1)

            # If receiving nodes is dirichlet or categorical, specify the target classes
            # TODO: Whether a new class is created could be determined by the polya urn model?
            if self.nodes[toI].type == NodeType.Dirichlet or self.nodes[toI].type == NodeType.Categorical:
                newEdge.toClass = self.nodes[toI].classCount

            # If the receiving node is a Gaussian that already has two inputs, declare the connection invalid
            if self.nodes[toI].type == NodeType.Gaussian:
                if self.nodes[toI].classCount >= 2:
                    invalid = True
                else:
                    # If the gaussian does not yet have a variance input (default is 1), then declare the connection valid
                    # and assign the variance index
                    newEdge.toClass = self.nodes[toI].classCount
                    self.nodes[toI].classCount += 1
                    invalid = False
            else:
                invalid = False

        # If the final receiving node is of type dirichlet, categorical or gaussian
        if (self.nodes[toI].type == NodeType.Dirichlet
                or self.nodes[toI].type == NodeType.Categorical
                or self.nodes[toI].type == NodeType.Gaussian):
            # Increase the class counter
            self.nodes[toI].classCount += 1

        # Finish adding the edge
        self.nodes[fromI].outputtingTo.append(toI)
        self.edges.append(newEdge)
        self.increaseLayers(self.nodes[fromI], self.nodes[toI])
        return 1

    # Replace an edge by two edges with a node in between
    def addNode(self, hMarker):
        # Search for an active edge to replace
        edge = self.edges[random.randint(0, len(self.edges)-1)]
        while(not edge.enabled):
            edge = self.edges[random.randint(0, len(self.edges) - 1)]

        # Add a node with random node type between the previously connected nodes
        node = AdvancedNodeGene.AdvancedNodeGene(nodeNr=len(self.nodes), type=NodeType.random())
        self.nodes.append(node)

        # Remove the old edge and
        # create two new edges that connect the previous two nodes via the new node
        self.specifiyEdge(edge, node, hMarker)

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(self, variance=1):
        indexWeight = random.randint(0, len(self.edges)-1)
        self.edges[indexWeight].weight = self.edges[indexWeight].weight + np.random.normal(0, variance)

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

        # If the new node is a dirichlet or categorical specify their input and output class indices for both edges
        if newNode.type == NodeType.Dirichlet or newNode.type == NodeType.Categorical:
            firstEdge.toClass = 0
            secondEdge.fromClass = 0

        # If new node is a gaussian specify the output class index for the new first new edge
        # For a gaussian this is the mean parameter.
        if newNode.type == NodeType.Gaussian:
            firstEdge.toClass = 0

        # Bookkeeping
        newNode.outputtingTo.append(edge.toNr)

        # Update the layer allocations for all affected nodes
        self.increaseLayers(self.nodes[edge.fromNr], newNode)

    # Implements the forward model for the SVI optimizer
    def model(self, data):
        # Get the nodes layer allocation for update sequencing
        layers = self.getLayers()

        # Splitting data into features and predictions
        inputData = data[0]
        outputData = data[1]

        # with pyro.plate("data", len(inputData)):
        # Loop through all but the last layer, as this one will be uniquely sampled with our data
        for i, layer in enumerate(layers):
            # If we are in the first layer, append our feature data to the inputs of the input nodes
            if(i == 0):
                for i2, nodeNr in enumerate(layer):
                    # print("InputData " + str(inputData[:,i2]))
                    self.nodes[nodeNr].inputs.append(inputData[:,i2])

            # If we are in the last layer, sample our output nodes predictions with our observations
            if (i == len(layers) - 1):
                for i, nodeNr in enumerate(layer):
                    if len(self.nodes[nodeNr].inputs) == 0:
                        self.nodes[nodeNr].inputs.append([torch.zeros(len(inputData), device=torch.device('cuda')),0])
                    with pyro.plate("obs_{}".format(i) + " plate", len(inputData)):
                        pyro.sample("obs_{}".format(i), self.nodes[nodeNr].function(), obs=outputData[:,i])
                continue

            # For each node in the layer calculate the output and distribute it to the other nodes
            for nodeNr in layer:
                # Calculate the output
                if not self.nodes[nodeNr].input and self.nodes[nodeNr].type != NodeType.Multiplication:
                    # and self.nodes[nodeNr].type != NodeType.Relu
                    with pyro.plate("node " + str(nodeNr) + " plate", len(inputData)):
                        output = pyro.sample("node " + str(nodeNr), self.nodes[nodeNr].function())
                        if(self.nodes[nodeNr].type == NodeType.Dirichlet
                                or self.nodes[nodeNr].type == NodeType.Categorical):
                            output = output.T
                else:
                    output = self.nodes[nodeNr].function()
                # print("output " + str(output))

                # Distribute weighted output to other nodes
                for nodeNr2 in list(dict.fromkeys(self.nodes[nodeNr].outputtingTo)):
                    for edge in self.edges:
                        if edge.enabled:
                            if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                if self.nodes[nodeNr].type != NodeType.Dirichlet \
                                        and self.nodes[nodeNr].type != NodeType.Categorical:
                                    # print("Inputting " + str(output * edge.weight))
                                    self.nodes[nodeNr2].inputs.append([output * edge.weight, edge.toClass])
                                else:
                                    self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])

    # Implements the model without observed variable and with hyperpriors
    def guide(self, data):
        # Get the nodes layer allocation for update sequencing
        layers = self.getLayers()

        # Splitting data into features and predictions
        inputData = data[0]
        outputData = data[1]

        # Sampling edge weights according to a normal prior
        for edge in self.edges:
            edge.weight = pyro.param("edge " + str(edge.fromNr) + " "
                                     + str(edge.toNr) + " "
                                     + str(edge.fromClass) + " "
                                     + str(edge.toClass), lambda: Normal(0, 1).sample([1]))\
                .to(torch.device('cuda'))

        # with pyro.plate("data", len(inputData)):
        # Loop through all but the last layer, as this one will be uniquely sampled with our data
        for i, layer in enumerate(layers):
            # If we are in the first layer, append our feature data to the inputs of the input nodes
            if(i == 0):
                for i2, nodeNr in enumerate(layer):
                    self.nodes[nodeNr].inputs.append(inputData[:,i2])

            # If we are in the last layer, sample our output nodes predictions with our observations
            if (i == len(layers) - 1):
                continue

            # For each node in the layer calculate the output and distribute it to the other nodes
            for nodeNr in layer:
                # Calculate the output
                if not self.nodes[nodeNr].input and self.nodes[nodeNr].type != NodeType.Multiplication:
                    #and self.nodes[nodeNr].type != NodeType.Relu
                    with pyro.plate("node " + str(nodeNr) + " plate", len(inputData)):
                        output = pyro.sample("node " + str(nodeNr), self.nodes[nodeNr].function())
                        if(self.nodes[nodeNr].type == NodeType.Dirichlet
                                or self.nodes[nodeNr].type == NodeType.Categorical):
                            output = output.T
                else:
                    output = self.nodes[nodeNr].function()
                # print(output.device)
                # Distribute weighted output to other nodes
                for nodeNr2 in list(dict.fromkeys(self.nodes[nodeNr].outputtingTo)):
                    for edge in self.edges:
                        if edge.enabled:
                            if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                # Ignore the previous weight in optimization
                                if self.nodes[nodeNr].type != NodeType.Dirichlet \
                                        and self.nodes[nodeNr].type != NodeType.Categorical:
                                    self.nodes[nodeNr2].inputs.append([output * edge.weight, edge.toClass])
                                else:
                                    self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])

    # Copies the current genome
    def copy(self):
        g = ProbabilisticGenome(0, 0)
        g.inputSize = self.inputSize
        g.outputSize = self.outputSize
        g.maxLayer = self.maxLayer

        g.edges = copy.deepcopy(self.edges)
        g.nodes = copy.deepcopy(self.nodes)
        g.fitness = self.fitness
        return g

    # Update all connected nodes' layer allocations
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
                # Receiving node may not be on a lower level than the sending node
                or (self.nodes[edge.toNr].layer <= self.nodes[edge.fromNr].layer)
                # Nodes may not send to themselves
                or edge.toNr == edge.fromNr):
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

    # Generates data given input
    def generate(self, inputData):
        # Get the nodes layer allocation for update sequencing
        layers = self.getLayers()

        # Looping over the inputData
        outputs = []
        # for input in inputData:
        # Loop through all but the last layer, as this one will be uniquely sampled with our data
        for i, layer in enumerate(layers):
            # If we are in the first layer, append our feature data to the inputs of the input nodes
            if(i == 0):
                for i2, nodeNr in enumerate(layer):
                    self.nodes[nodeNr].inputs.append(inputData[:,i2])

            # If we are in the last layer, sample our output nodes predictions with our observations
            if (i == len(layers) - 1):
                for nodeNr in layer:
                    if len(self.nodes[nodeNr].inputs) == 0:
                        self.nodes[nodeNr].inputs.append([torch.zeros(len(inputData), device=torch.device('cuda')),0])
                    if not self.nodes[nodeNr].input and self.nodes[nodeNr].type != NodeType.Multiplication:
                        # and self.nodes[nodeNr].type != NodeType.Relu
                        y = self.nodes[nodeNr].function().sample([1])[0]
                    else:
                        y = self.nodes[nodeNr].function()
                    torch.reshape(y, (len(y), -1))
                    outputs.append(y)

                outputs = torch.stack(outputs, dim=1)
                continue

            # For each node in the layer calculate the output and distribute it to the other nodes
            for nodeNr in layer:
                # Calculate the output
                if not self.nodes[nodeNr].input and self.nodes[nodeNr].type != NodeType.Multiplication:
                    #self.nodes[nodeNr].type != NodeType.Relu and
                    output = self.nodes[nodeNr].function().sample([1])[0]
                    if(self.nodes[nodeNr].type == NodeType.Dirichlet or self.nodes[nodeNr].type == NodeType.Categorical):
                        output = output.T


                else:
                    output = self.nodes[nodeNr].function()
                # Distribute weighted output to other nodes
                for nodeNr2 in list(dict.fromkeys(self.nodes[nodeNr].outputtingTo)):
                    # Dirichlet and Categorical distributions have a different output format and must be handled differently
                    if self.nodes[nodeNr].type != NodeType.Dirichlet \
                            and self.nodes[nodeNr].type != NodeType.Categorical:
                        for edge in self.edges:
                            if edge.enabled:
                                if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                    self.nodes[nodeNr2].inputs.append([output * edge.weight, edge.toClass])
                    else:
                        for edge in self.edges:
                            if edge.enabled:
                                if(edge.fromNr == nodeNr and edge.toNr == nodeNr2):
                                    self.nodes[nodeNr2].inputs.append([output[edge.fromClass] * edge.weight, edge.toClass])
        return outputs

    # Returns how many of each nodetype are in the current genome
    def nodeStats(self):
        dirichlets = 0
        categoricals = 0
        gaussians = 0
        relus = 0
        multiplications = 0
        for node in self.nodes:
            if node.input:
                continue
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

    # Visualize the genome's graph
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


        # Add nodes to network
        labels = {}
        for node, nodePos in zip(nodes, nodePositions):
            G.addNode(node, pos=(nodePos[1], -nodePos[0]))
            if(self.nodes[node].input):
                labels[node] = "inp"
            else:
                labels[node] = str(self.nodes[node].type) + " " + str(node)
                if self.nodes[node].output:
                    labels[node] = labels[node] + "\n out"
                # Additional nodes for distributions for visualization
                if self.nodes[node].type == NodeType.Dirichlet or self.nodes[node].type == NodeType.Categorical:
                    for i in range(self.nodes[node].classCount):
                        pos = (nodePos[0] - 0.3,
                               nodePos[1] + i*0.4/self.nodes[node].classCount - 0.1*(max(0, min(1, self.nodes[node].classCount-1))))
                        G.addNode(str(node) + " in " + str(i), pos=(pos[1], -pos[0]))
                        labels[str(node) + " in " + str(i)] = "c" + str(i)

                if self.nodes[node].type == NodeType.Dirichlet or self.nodes[node].type == NodeType.Categorical:
                    for i in range(self.nodes[node].classCount):
                        pos = (nodePos[0] + 0.3,
                               nodePos[1] + i*0.4/self.nodes[node].classCount - 0.1*(max(0, min(1, self.nodes[node].classCount-1))))
                        G.addNode(str(node) + " out " + str(i), pos=(pos[1], -pos[0]))
                        labels[str(node) + " out " + str(i)] = "c" + str(i)

                if self.nodes[node].type == NodeType.Gaussian:
                    for i in range(self.nodes[node].classCount):
                        if(i > 1):
                            break
                        pos = (nodePos[0] - 0.3,
                               nodePos[1] + i*0.4/self.nodes[node].classCount - 0.1*(max(0, min(1, self.nodes[node].classCount-1))))
                        G.addNode(str(node) + " in " + str(i), pos=(pos[1], -pos[0]))
                        if i == 0:
                            labels[str(node) + " in " + str(i)] = "u"
                        else:
                            labels[str(node) + " in " + str(i)] = "var"



        edgeLabels = []
        edgeRads = []
        for edge in self.edges:
            if(edge.enabled):
                # Special senders and special receivers
                if (self.nodes[edge.toNr].type == NodeType.Dirichlet
                        or self.nodes[edge.toNr].type == NodeType.Categorical
                        or self.nodes[edge.toNr].type == NodeType.Gaussian) \
                    and (self.nodes[edge.fromNr].type == NodeType.Dirichlet
                         or self.nodes[edge.fromNr].type == NodeType.Categorical):
                    # From sending node to output class node
                    G.addEdge(edge.fromNr, str(edge.fromNr) + " out " + str(edge.fromClass))

                    # From output class node to receiving input class node
                    G.addEdge(str(edge.fromNr) + " out " + str(edge.fromClass), str(edge.toNr) + " in " + str(edge.toClass))
                    # Label with weight
                    edgeLabels.append(((str(edge.fromNr) + " out " + str(edge.fromClass), str(edge.toNr) + " in " + str(edge.toClass)), round(edge.weight, 3)))

                    # From receiving input class node to receiving main node
                    G.addEdge(str(edge.toNr) + " in " + str(edge.toClass), edge.toNr)

                else:
                    # Just special receivers
                    if self.nodes[edge.toNr].type == NodeType.Dirichlet \
                            or self.nodes[edge.toNr].type == NodeType.Categorical \
                            or self.nodes[edge.toNr].type == NodeType.Gaussian:
                        G.addEdge(edge.fromNr, str(edge.toNr) + " in " + str(edge.toClass))
                        G.addEdge(str(edge.toNr) + " in " + str(edge.toClass), edge.toNr)
                        edgeLabels.append(((edge.fromNr, str(edge.toNr) + " in " + str(edge.toClass)), round(edge.weight, 3)))
                    # Just special senders
                    elif self.nodes[edge.fromNr].type == NodeType.Dirichlet \
                                or self.nodes[edge.fromNr].type == NodeType.Categorical:
                        G.addEdge(edge.fromNr, str(edge.fromNr) + " out " + str(edge.fromClass))
                        G.addEdge(str(edge.fromNr) + " out " + str(edge.fromClass), edge.toNr)
                        edgeLabels.append(((str(edge.fromNr) + " out " + str(edge.fromClass), edge.toNr), round(edge.weight, 3)))
                    else:
                        # Nothing special
                        G.addEdge(edge.fromNr, edge.toNr, rad=((self.nodes[edge.toNr].layer - self.nodes[edge.fromNr].layer)-1)*0.1)
                        edgeLabels.append(((edge.fromNr, edge.toNr), round(edge.weight, 3)))

                edgeRads.append(((edge.fromNr, edge.toNr), (self.nodes[edge.toNr].layer - self.nodes[edge.fromNr].layer)))
                parents[nodes.index(edge.toNr)].append(edge.fromNr)

        G.visualize(ion= ion, labels=labels, edgeLabels=dict(edgeLabels))

    # Trains the genome using SVI
    def train(self, data, num_iterations, optim = Adam({"lr": 0.05}), loss= Trace_ELBO()):
        svi = SVI(self.model, self.guide, optim, loss=loss)

        pyro.set_rng_seed(0)
        pyro.clear_param_store()

        losses = []
        torch.autograd.set_detect_anomaly(True)
        for j in tqdm(range(num_iterations)):
            loss = svi.step(data)
            losses.append(loss)

        print(pyro.get_param_store().keys())

        for edge in self.edges:
            edge.weight = pyro.param("edge " + str(edge.fromNr) + " "
                                     + str(edge.toNr) + " "
                                     + str(edge.fromClass) + " "
                                     + str(edge.toClass)).detach().item()

        return losses, losses[len(losses)-1]

    # Returns a list of lists in which all raw data of this genome is stored
    def toData(self):
        data = [self.inputSize, self.outputSize, self.maxLayer]

        nodeData = []
        edgeData = []
        for node in self.nodes:
            nodeData.append(node.toData())
        for edge in self.edges:
            edgeData.append(edge.toData())

    @staticmethod
    # Builds a genome from raw data
    def fromData(data):
        nodes = []
        edges = []

        inputCount, outputCount, maxLayer = 0, 0, 0
        for node in data[0]:
            nodes.append(AdvancedNodeGene.AdvancedNodeGene.fromData(node))
            if node.input:
                inputCount += 1
            elif node.output:
                outputCount += 1
            maxLayer = max(maxLayer, node.layer)

        for edge in data[1]:
            edges.append(AdvancedEdgeGene.AdvancedEdgeGene.fromData(edge))

        genome = ProbabilisticGenome(0, 0)
        genome.inputSize = inputCount
        genome.outputSize = outputCount
        genome.maxLayer = maxLayer
        genome.nodes = nodes
        genome.edges = edges

        return genome