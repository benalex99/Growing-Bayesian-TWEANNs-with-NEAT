import ast
import csv
import os
import random
import shutil
from os.path import exists

import numpy as np
from tqdm import tqdm

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.NEATEnv import VariableEnv
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticNEAT import ProbabilisticNEAT
import matplotlib.pyplot as plt
import torch
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.StructuralDistance import DistanceMetric

class Analysis():

    @staticmethod
    def NeatVsNormalTest(genomeSampleSize=10000, samples=1000,
                         distributionSampleSize=1000, detail=20,
                         optimizerIters=40, maxPopSize=100, batchSize=10,
                         dataIndex=0, genomes=None):
        '''
        Calculates KL-divergence vs structural distance.
        :param genomeSampleSize: The amount of genomes to perform tests on
        :param samples: The amount of genome pairs per genome

        :param distributionSampleSize: The amount of samples to draw from a distribution to calculate the KL-divergence
        :param detail: The detail with which the distributions are converted to discrete distributions

        :param optimizerIters: The amount of iterations to run the optimizer, before calculating the KL-divergence
        :param maxPopSize: The maximum amount of candidates kept in the population
        :param batchSize: The amount of candidates generated for each iteration
        :param dataIndex: The name of the folder in which the data of this run will be stored

        :param genomes: A list of genomes, that will be used for analysis if provided.
        :return: Stores resulting data in CSV files.
        '''
        # Create a root folder with a new number as name.
        rootPath = "Data\\" + str(dataIndex)
        while exists(rootPath):
            dataIndex += 1
            rootPath = "Data\\" + str(dataIndex)
        os.mkdir(rootPath)

        # If no genomes were provided, create our own and store them in the root folder.
        if genomes is None:
            genomes, newGenomes = Analysis.generateGenomes(genomeSampleSize)
            Analysis.storeGenomesInCsv(genomes, rootPath=rootPath)

        # Create random genome pairings to test
        indices = Analysis.generatePairs(len(genomes), samples)[:,0]

        distanceData = []
        divergenceData = []
        lossData = []
        KLDivs = []
        # Generate our test results
        for index in tqdm(indices, desc="Calculating data", unit="pairs"):
            Analysis.NeatTest()

            # Perform the tests on all genome pairs
            distances = Analysis.distanceTest(genomes, pair)
            divergences, losses, KLDiv = Analysis.KLDivTest(genomes, pair, sampleSize=distributionSampleSize, detail=detail,
                                                            iterations=optimizerIters, maxPopSize=maxPopSize, batchSize=batchSize)

            # Update our data store
            distanceData.append(distances)
            divergenceData.append(divergences)
            lossData.append(losses)
            KLDivs.append(KLDiv)
            Analysis.storeDataInCsv(genomePairs, distanceData, divergenceData, lossData, KLDivs, rootPath=rootPath)




    @staticmethod
    def KLvsDistance(genomeSampleSize=10000, samples=1000,
                     distributionSampleSize=1000, detail=20,
                     optimizerIters=40, maxPopSize=100, batchSize=10,
                     dataIndex=0, genomes=None):
        '''
        Calculates KL-divergence vs structural distance.
        :param genomeSampleSize: The amount of genomes to perform tests on
        :param samples: The amount of genome pairs per genome

        :param distributionSampleSize: The amount of samples to draw from a distribution to calculate the KL-divergence
        :param detail: The detail with which the distributions are converted to discrete distributions

        :param optimizerIters: The amount of iterations to run the optimizer, before calculating the KL-divergence
        :param maxPopSize: The maximum amount of candidates kept in the population
        :param batchSize: The amount of candidates generated for each iteration
        :param dataIndex: The name of the folder in which the data of this run will be stored

        :param genomes: A list of genomes, that will be used for analysis if provided.
        :return: Stores resulting data in CSV files.
        '''
        # Create a root folder with a new number as name.
        rootPath = "Data\\" + str(dataIndex)
        while exists(rootPath):
            dataIndex += 1
            rootPath = "Data\\" + str(dataIndex)
        os.mkdir(rootPath)

        # If no genomes were provided, create our own and store them in the root folder.
        if genomes is None:
            genomes, newGenomes = Analysis.generateGenomes(genomeSampleSize)
            Analysis.storeGenomesInCsv(genomes, rootPath=rootPath)

        # Create random genome pairings to test
        genomePairs = Analysis.generatePairs(len(genomes), samples)

        distanceData = []
        divergenceData = []
        lossData = []
        KLDivs = []
        # Generate our test results
        for pair in tqdm(genomePairs, desc="Calculating data", unit="pairs"):
            # Perform the tests on all genome pairs
            distances = Analysis.distanceTest(genomes, pair)
            divergences, losses, KLDiv = Analysis.KLDivTest(genomes, pair, sampleSize=distributionSampleSize, detail=detail,
                                             iterations=optimizerIters, maxPopSize=maxPopSize, batchSize=batchSize)

            # Update our data store
            distanceData.append(distances)
            divergenceData.append(divergences)
            lossData.append(losses)
            KLDivs.append(KLDiv)
            Analysis.storeDataInCsv(genomePairs, distanceData, divergenceData, lossData, KLDivs, rootPath=rootPath)

    @staticmethod
    def plotData():
        path = "Data\\1"
        allPairs, allDistances, allDivergences, allLossData, allKLDivs = [], [], [], [], []

        pairs, distances, divergences, lossData, KLDivs = Analysis.retrieveDataFromCsv(path)
        allPairs.extend(pairs)
        allDistances.extend(distances)
        allDivergences.extend(divergences)
        allLossData.extend(lossData)
        allKLDivs.extend(KLDivs)

        path = "Data\\2"
        pairs, distances, divergences, lossData, KLDivs = Analysis.retrieveDataFromCsv(path)
        allPairs.extend(pairs)
        allDistances.extend(distances)
        allDivergences.extend(divergences)
        allLossData.extend(lossData)
        allKLDivs.extend(KLDivs)

        for i in range(len(allDivergences)):
            allDivergences[i] *= -1

        for i in range(len(allLossData)):
            allLossData[i] = np.array(allLossData[i]) * -1

        weights = np.arange(0, 40)
        allLossData.sort(key=lambda x: sum(x*weights), reverse=False)

        # Plot loss data ordered by structural distance
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(18.5, 10.5)

        for x, (listo, dist) in enumerate(zip(allLossData, allDistances)):
            ax.plot3D(np.ones(40)*dist, np.arange(0, 40), np.array(listo),
                             linewidth=2, antialiased=False)
        plt.show()

        # Plot loss data ordered by weighted loss. Creates a nice surface to showcase the frequency of loss trajectories
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(18.5, 10.5)

        for x, (listo, dist) in enumerate(zip(allLossData, allDistances)):
            ax.plot3D(np.ones(40)*x, np.arange(0, 40), np.array(listo),
                      linewidth=2, antialiased=False)

        plt.show()

        # Plot the final loss vs structural distance
        plt.scatter(x=allDistances, y=allDivergences)
        xMin, xMax = min(allDistances), max(allDistances)
        yMin, yMax = min(allDivergences), max(allDivergences)
        plt.xticks([xMin, xMax])
        plt.yticks([yMin, yMax])
        plt.show()

    @staticmethod
    def distanceTest(genomes, pair):
        genome = genomes[pair[0]]
        otherGenome = genomes[pair[1]]
        return DistanceMetric.run(genome, otherGenome)

    @staticmethod
    def KLDivTest(genomes, pair, sampleSize=1000, detail=20, iterations=10, maxPopSize=200, batchSize=100,):
        env = VariableEnv(genomes[pair[0]], datapointCount=sampleSize, detail=detail)
        otherGenome = genomes[pair[1]]

        optimizer = ProbabilisticNEAT(iterations,
                                      maxPopSize=maxPopSize, batchSize=batchSize,
                                      weightsOnly=True, useMerging=False, useSpeciation=False)
        otherGenome, losses = optimizer.run(otherGenome, env)
        KLDiv = env.discretizedKullbackLeibler(env.generated, otherGenome.generate(env.input))
        return otherGenome.fitness, losses, KLDiv

    @staticmethod
    def generateGenomes(N, input=1, output=1, startPop=None):
        if startPop == None or len(startPop) == 0:
            genomes = [ProbabilisticGenome(inputSize=input, outputSize=output)]
        else:
            genomes = startPop
        newGenomes = []
        hMarker = 0
        for _ in tqdm(range(N), desc="Generating genomes", unit="genome"):
            success = 0
            parent = genomes[random.randint(0,len(genomes)-1)]
            newGenome = parent.copy()
            while(not success > 0):
                success = newGenome.mutate(hMarker)
            hMarker += success
            genomes.append(newGenome)
            newGenomes.append(newGenome)
        return genomes, newGenomes

    @staticmethod
    def generatePairs(genomeCount, sampleCount):
        pairs = []
        for _ in range(sampleCount):
            pairs.append([random.randint(0, genomeCount-1), random.randint(0, genomeCount-1)])
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
        print(alignments)
        alignments, indices = torch.max(alignments, dim=0)
        print(alignments)
        return (len(genome1.nodes)-len(genome2.nodes) + 1 - (torch.sum(torch.sum(alignments))/((len(genome1.nodes)+len(genome1.nodes))/2)))

    @staticmethod
    def storeGenomesInCsv(genomes, rootPath="Data"):
        rootPath = rootPath + "\\Genomes"
        if not exists(rootPath):
            os.mkdir(rootPath)
        else:
            shutil.rmtree(rootPath)
            os.mkdir(rootPath)

        for i, genome in enumerate(genomes):

            # Store the genomes nodes
            path = rootPath +"\\" + str(i)
            os.mkdir(path)
            path += "\\nodes.csv"
            with open(path, mode='w') as file:
                data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(8))
                for node in genome.nodes:
                    data_writer.writerow(dict(zip(range(8),node.toData())))

            # Store the genomes edges
            path = rootPath + "\\" +str(i) + "\\edges.csv"
            with open(path, mode='w') as file:
                data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(8))
                for node in genome.edges:
                    data_writer.writerow(dict(zip(range(8),node.toData())))

    @staticmethod
    def readGenomesFromCsv(rootPath="Data"):
        rootPath = rootPath + "\\Genomes"
        if not exists(rootPath):
            os.mkdir(rootPath)
            return []

        genomeFolders = os.listdir(rootPath)
        genomeFolders.sort(key=lambda x: int(x), reverse=False)

        genomes = []
        for Folder in tqdm (genomeFolders, desc="Loading genomes", unit="G"):
            nodeData = []
            edgeData = []

            # Retrieve the genomes nodes
            path = rootPath + "\\" + Folder + "\\nodes.csv"
            with open(path, mode='r') as file:
                data_reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(8))
                for row in data_reader:
                    nodeData.append(row)

            # Retrieve the genomes edges
            path = rootPath + "\\" + Folder + "\\edges.csv"
            with open(path, mode='r') as file:
                data_reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=range(8))
                for row in data_reader:
                    edgeData.append(row)

            genomes.append(ProbabilisticGenome.fromData([nodeData, edgeData]))
        return genomes

    @staticmethod
    def storeDataInCsv(pairs, distances, divergences, lossData, KLDivs, rootPath="Data"):
        path = rootPath + "\\Data.csv"

        fieldnames = ["Env", "Model", "Structural Distance", "KL-Divergence", "Losses", "KLDiv"]
        with open(path, mode='w') as file:
            data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                         fieldnames=fieldnames)
            for (i, compareTo), distance, divergence, losses, KLDiv in \
                    zip(pairs, distances, divergences, lossData, KLDivs):
                data_writer.writerow({"Env": i,
                                      "Model": compareTo,
                                      "Structural Distance": distance,
                                      "KL-Divergence": divergence,
                                      "Losses": losses,
                                      "KLDiv": KLDiv})

    @staticmethod
    def retrieveDataFromCsv(rootPath="Data"):
        path = rootPath + "\\Data.csv"
        pairs = []
        distances = []
        divergences = []
        lossData = []
        KLDivs = []
        fieldnames = ["Env", "Model", "Structural Distance", "KL-Divergence", "Losses", "KLDiv"]
        if exists(path):
            with open(path, mode='r') as file:
                data_reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=fieldnames)
                for row in tqdm(data_reader, desc= "Loading Data from CSV", unit= "rows"):
                    pairs.append([ast.literal_eval(row["Env"]), ast.literal_eval(row["Model"])])
                    distances.append(ast.literal_eval(row["Structural Distance"]))
                    divergences.append(ast.literal_eval(row["KL-Divergence"]))
                    lossData.append(ast.literal_eval(row["Losses"]))
                    if row["KLDiv"] is not None:
                        KLDivs.append(ast.literal_eval(row["KLDiv"]))
        return pairs, distances, divergences, lossData, KLDivs
