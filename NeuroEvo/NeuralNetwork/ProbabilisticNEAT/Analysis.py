import csv
import os
import random

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.NEATEnv import VariableEnv
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticNEAT import ProbabilisticNEAT
import matplotlib.pyplot as plt
import torch

class Analysis():

    @staticmethod
    def KLvsDistance(genomeCount, samplesPerGenome, sampleSize = 1000):
        genomes = Analysis.generateGenomes(genomeCount)
        Analysis.storeGenomesInCsv(genomes)

        genomePairs = Analysis.generatePairs(genomeCount, samplesPerGenome)
        distanceData = []
        divergenceData = []
        for i, pairs in enumerate(genomePairs):
            distances = Analysis.distanceTest(i, genomes, pairs)
            divergences = Analysis.KLDivTest(i, genomes, pairs, sampleSize=sampleSize)

            Analysis.storeDataInCsv(pairs, distances, divergences)

            distanceData.append(distances)
            divergenceData.append(divergences)

        Analysis.plotData(distanceData, divergenceData)

    @staticmethod
    def plotData(distanceData, divergenceData):
        plt.scatter(distanceData, divergenceData)

    @staticmethod
    def distanceTest(index, genomes, pairs):
        distances = []

        genome = genomes[index]
        for i in pairs:
            otherGenome = genomes[i]
            distance = genome.compareStructure(otherGenome)
            distances.append(distance)
        return distances

    @staticmethod
    def KLDivTest(index, genomes, pairs, sampleSize = 1000, iterations = None):
        divergences = []

        genome = genomes[index]
        env = VariableEnv(genome, datapointCount=sampleSize)
        for i in pairs:
            otherGenome = genomes[i]
            # If no iteration count is given, use the amount of edges of the to be optimized genome.
            # TODO: Optional: have possible that each weight is updated at least once
            if iterations == None:
                iterations = len(otherGenome.edges)

            optimizer = ProbabilisticNEAT(iterations,
                                          maxPopSize=200, batchSize=100, episodeDur=0,
                                          weightsOnly=True, useMerging=False, useSpeciation=False)
            otherGenome = optimizer.run(env, otherGenome)
            # Fitness is calculated with KL-divergence
            divergences.append(otherGenome.fitness)
        return divergences

    @staticmethod
    def generateGenomes(N, input=1, output=1):
        genomes = [ProbabilisticGenome(inputSize=input, outputSize=output)]
        hMarker = 0
        for _ in N:
            parent = genomes[random.randint(0,len(genomes))]
            newGenome = parent.copy()
            hMarker += newGenome.mutate(hMarker)
            genomes.append(newGenome)
        return genomes

    @staticmethod
    def generatePairs(genomeCount, samplesPerGenome):
        pairs = []
        for genome in range(genomeCount):
            indices = []
            for _ in range(samplesPerGenome):
                indices.append(random.randint(0, genomeCount-1))
            pairs.append(indices)
        return pairs

    @staticmethod
    def structuralDivergence(genome1, genome2, typeMatters=True):
        if len(genome1.nodes) < len(genome2.nodes):
            help = genome1
            genome1 = genome2
            genome2 = help
        layers = genome1.getLayers()
        otherLayers = genome2.getLayers()

        # Assign the edges to their respective nodes for performances sake
        for node in genome1.nodes:
            node.edges = []
        for edge in genome1.edges:
            if edge.enabled:
                genome1.nodes[edge.toNr].edges.append(edge)
        for node in genome2.nodes:
            node.edges = []
        for edge in genome2.edges:
            if edge.enabled:
                genome2.nodes[edge.toNr].edges.append(edge)

        # Initialize node alignments with 0
        alignments = torch.zeros((len(genome1.nodes), len(genome2.nodes)))
        # Set the alignments of the input nodes to 1 (Assuming we have equal inputsizes and inputs)
        alignments[:genome1.inputSize, :genome2.inputSize] = 1
        for i, layer in enumerate(layers):
            if i == 0:
                continue
            for i2, layer2 in enumerate(otherLayers):
                if i2 == 0:
                    continue
                for node in layer:
                    for node2 in layer2:
                        if typeMatters and genome1.nodes[node].type != genome2.nodes[node2].type:
                            continue

                        for edge in genome1.nodes[node].edges:
                            bestAlignment = 0
                            for edge2 in genome2.nodes[node2].edges:
                                if alignments[edge.fromNr, edge2.fromNr] > bestAlignment:
                                    bestAlignment = alignments[edge.fromNr, edge2.fromNr]
                            alignments[node, node2] += bestAlignment

                        alignments[node, node2] /= max(len(genome1.nodes[node].edges), len(genome2.nodes[node2].edges))

                        # if len(self.nodes[node].edges) + len(other.nodes[node2].edges) > 0:
                        #     print("Nodes: " + str(node) + " " + str(node2) +
                        #           " Edges: " + str(len(self.nodes[node].edges)) + " " + str(len(other.nodes[node2].edges)) +
                        #           " Alignment: " + str(alignments[node, node2].item()))
                        #     for edge in self.nodes[node].edges:
                        #         for edge2 in other.nodes[node2].edges:
                        #             print("edge " + str(edge.fromNr) + " " + str(edge2.fromNr) + " "
                        #                   + str(alignments[edge.fromNr, edge2.fromNr].item()))
        print(alignments)
        alignments, indices = torch.max(alignments, dim=0)
        print(alignments)
        return (len(genome1.nodes)-len(genome2.nodes) + 1 - (torch.sum(torch.sum(alignments))/((len(genome1.nodes)+len(genome1.nodes))/2)))

    @staticmethod
    def addToCsv(data, path="data.txt"):
        # Determine how many genomes weve already done
        with open(path, 'r') as file:
            reader = csv.reader(file)
            rowCount = sum(1 for row in reader)

        # Write the structural distances and KL divergences, with the index of their respective genome structure
        with open(path, 'a') as file:
            writer = csv.writer(file)
            for [dist,div] in data:
                writer.writerow([dist,div])

    @staticmethod
    def storeGenomesInCsv(genomes, rootPath="Genomes\\"):
        for i, genome in enumerate(genomes):

            # Store the genomes nodes
            path = rootPath + str(i) + "\\nodes.csv"
            with open(path, mode='w') as file:
                data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(7))
                for node in genome.nodes:
                    data_writer.writerow(node.toData())

            # Store the genomes edges
            path = rootPath + str(i) + "\\edges.csv"
            with open(path, mode='w') as file:
                data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(7))
                for node in genome.nodes:
                    data_writer.writerow(node.toData())

    @staticmethod
    def readGenomesFromCsv(rootPath="Genomes\\"):
        genomeFolders = os.listdir()
        genomeFolders.sort(key=lambda x: int(x), reverse=True)

        genomes = []
        for Folder in genomeFolders:
            nodeData = []
            edgeData = []

            # Retrieve the genomes nodes
            path = rootPath + Folder + "\\nodes.csv"
            with open(path, mode='r') as file:
                data_reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(7))
                for row in data_reader:
                    nodeData.append(row)

            # Retrieve the genomes edges
            path = rootPath + Folder + "\\edges.csv"
            with open(path, mode='r') as file:
                data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(7))
                for row in data_reader:
                    edgeData.append(row)

            genomes.append(ProbabilisticGenome.fromData([nodeData, edgeData]))
        return genomes

    @staticmethod
    def storeDataInCsv(pairs, distances, divergences, path="Data.csv"):
        fieldnames = ["Env", "Model", "Structural Distance", "KL-Divergence"]
        with open(path, mode='w') as file:
            data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                         fieldnames=fieldnames)
            for [env, model], distance, divergence in zip(pairs, distances, divergences):
                data_writer.writerow({"Env": env, "Model": model,
                                      "Structural Distance": distance, "KL-Divergence": divergence})

    @staticmethod
    def retrieveDataFromCsv(path="Data.csv"):
        data = []
        with open(path, mode='a') as file:
            data_reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for row in data_reader:
                data.append([row["Env"], row["Model"], row["Structural Distance"], row["KL-Divergence"]])
        return data