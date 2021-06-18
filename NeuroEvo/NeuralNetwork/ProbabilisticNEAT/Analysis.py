import ast
import csv
import os
import random
import shutil
from os.path import exists
from sklearn.linear_model import LinearRegression

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
                         dataIndex=0, genomes=None,
                         useMergingAndSpeciation=True,
                         criterion="me"):
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
        rootPath = "Data\\NEATTest\\" + criterion + "\\" + str(useMergingAndSpeciation) + "\\" + str(dataIndex)
        while exists(rootPath):
            dataIndex += 1
            rootPath = "Data\\NEATTest\\" + criterion + "\\" + str(useMergingAndSpeciation) + "\\" + str(dataIndex)
        os.mkdir(rootPath)

        # If no genomes were provided, create our own and store them in the root folder.
        if genomes is None:
            genomes, newGenomes = Analysis.generateGenomes(genomeSampleSize)
            Analysis.storeGenomesInCsv(genomes, rootPath=rootPath)

        # Create random genome pairings to test
        indices = (np.array(Analysis.generatePairs(len(genomes), samples))[:, 0]).tolist()

        distanceData = []
        divergenceData = []
        lossData = []
        KLDivs = []
        # Generate our test results
        for index in tqdm(indices, desc="Calculating data", unit="tests"):

            # Perform the tests on all genome pairs
            divergences, losses, KLDiv, distances = Analysis.NeatTest(genomes, index,
                                                                      sampleSize=distributionSampleSize, detail=detail,
                                                            iterations=optimizerIters, maxPopSize=maxPopSize, batchSize=batchSize,
                                                            useMergingAndSpeciation=useMergingAndSpeciation,
                                                                      criterion="me")

            # Update our data store
            distanceData.append(distances)
            divergenceData.append(divergences)
            lossData.append(losses)
            KLDivs.append(KLDiv)

            ind = zip(indices, np.ones(len(indices)) * -1)
            Analysis.storeDataInCsv(ind, distanceData, divergenceData, lossData, KLDivs, rootPath=rootPath)


    @staticmethod
    def plotNeatTestData(withSpeciationAndMerging = True, dataset=0):
        path = "Data\\NeatTest\\me\\" + str(withSpeciationAndMerging) + "\\" + str(dataset)
        allPairs, allDistances, allFinalLosses, allLossData, allKLDivs = [], [], [], [], []

        pairs, distances, divergences, lossData, KLDivs = Analysis.retrieveDataFromCsv(path)
        allPairs.extend(pairs)
        allDistances.extend(distances)
        allFinalLosses.extend(divergences)
        allLossData.extend(lossData)
        optimIterations = len(allLossData[0])
        allKLDivs.extend(KLDivs)

        # For both all losses invert the sign. Our optimizers maximize fitnesses, so we gave it the negative loss.
        # To properly plot the loss, we have to revert it back.
        for i in range(len(allFinalLosses)):
            allFinalLosses[i] *= -1

        for i in range(len(allLossData)):
            allLossData[i] = np.array(allLossData[i]) * -1

        weights = np.arange(0, optimIterations)
        allLossData = np.array(allLossData)
        allDistances = np.array(allDistances)
        allLossData2 = np.sum(allLossData * weights, axis=1)
        indices = np.argsort(allLossData2)

        allLossData = allLossData.take(indices, 0)
        allDistances = allDistances.take(indices, 0)
        allDistances2 = np.repeat(allDistances, 4, axis=1)
        allDistances2 = allDistances2[:, 0:optimIterations]

        # Plot loss data
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        # fig.set_size_inches(18.5, 10.5)
        #
        # for x, (listo, dist) in enumerate(zip(allLossData, allDistances2)):
        #     ax.plot3D(np.array(dist), np.arange(0, optimIterations), np.array(listo),
        #               linewidth=2, antialiased=False)
        # ax.set_xlabel("Structural distance")
        # ax.set_ylabel("Iteration")
        # ax.set_zlabel("Loss")
        # plt.show()

        # Plot loss data ordered by weighted loss. Creates a nice surface to showcase the frequency of loss trajectories
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(18.5, 10.5)

        for x, listo in enumerate(allLossData):
            ax.plot3D(np.ones(optimIterations)*x, np.arange(0, optimIterations), np.array(listo),
                      linewidth=2, antialiased=False)
        ax.set_xlabel("Sorted by area under the curve")
        ax.set_ylabel("Iteration")
        ax.set_zlabel("Loss")
        plt.show()

        # Plot the average loss over time
        medianLoss = []
        percentile25 = []
        percentile75 = []
        for i in range(len(allLossData[0])):
            medianLoss.append(np.percentile(allLossData[:, i], 50))
            percentile25.append(np.percentile(allLossData[:, i], 25))
            percentile75.append(np.percentile(allLossData[:, i], 75))
        plt.plot(np.arange(len(allLossData[0])), medianLoss, label="median loss")
        plt.plot(np.arange(len(allLossData[0])), percentile25, color="black", label="25 and 75 percentile")
        plt.plot(np.arange(len(allLossData[0])), percentile75, color="black")

        plt.xlabel("Iteration")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        # Plot the average structural distance over time
        medianDistance = []
        percentile25 = []
        percentile75 = []
        for i in range(len(allDistances2[0])):
            medianDistance.append(np.percentile(allDistances2[:, i], 50))
            percentile25.append(np.percentile(allDistances2[:, i], 25))
            percentile75.append(np.percentile(allDistances2[:, i], 75))
        plt.plot(np.arange(len(allDistances2[0])), medianDistance, label="median loss")
        plt.plot(np.arange(len(allDistances2[0])), percentile25, color="black", label="25 and 75 percentile")
        plt.plot(np.arange(len(allDistances2[0])), percentile75, color="black")

        plt.xlabel("Iteration")
        plt.ylabel("structural distance")
        plt.legend()
        plt.show()


        # Plot the final loss vs structural distance
        finalDistances = allDistances2[:, optimIterations-1]
        plt.scatter(x=finalDistances, y=allFinalLosses)
        xMin, xMax = min(finalDistances), max(finalDistances)
        yMin, yMax = min(allFinalLosses), max(allFinalLosses)
        plt.xticks([xMin, xMax])
        plt.yticks([yMin, yMax])
        plt.show()

    @staticmethod
    def BaseLineVsNEATLoss():
        path = "Data\\NeatTest\\me\\False\\1"
        _, _, _, baseLossData, _ = Analysis.retrieveDataFromCsv(path)
        baseLossData = np.array(baseLossData)

        path = "Data\\NeatTest\\me\\True\\1"
        _, _, _, NEATLossData, _ = Analysis.retrieveDataFromCsv(path)
        NEATLossData = np.array(NEATLossData)

        # Invert losses. We converted loss to fitness, as both optimizers maximize fitness. Now we revert back to loss.
        baseLossData *= -1
        NEATLossData *= -1

        # Remove models that started out already perfectly matching
        baseLossData = baseLossData[baseLossData[:, 0] != 0]
        NEATLossData = NEATLossData[NEATLossData[:, 0] != 0]

        fig, ax = plt.subplots(1,2, sharey="all")
        fig.set_size_inches(18.5/2, 10.5/2)

        # Plot the average loss over time
        meanLoss = []
        percentile0 = []
        percentile25 = []
        percentile75 = []
        percentile100 = []
        for i in range(len(baseLossData[0])):
            meanLoss.append(np.mean(baseLossData[:, i]))
            percentile0.append(np.percentile(baseLossData[:, i], 0))
            percentile25.append(np.percentile(baseLossData[:, i], 25))
            percentile75.append(np.percentile(baseLossData[:, i], 75))
            percentile100.append(np.percentile(baseLossData[:, i], 100))
        ax[0].plot(np.arange(len(baseLossData[0])), meanLoss, label="mean")
        ax[0].plot(np.arange(len(baseLossData[0])), percentile0, color="red", label="0th and 100th percentile")
        ax[0].plot(np.arange(len(baseLossData[0])), percentile25, color="black", label="25th and 75th percentile")
        ax[0].plot(np.arange(len(baseLossData[0])), percentile75, color="black")
        ax[0].plot(np.arange(len(baseLossData[0])), percentile100, color="red")
        ax[0].set_title("Basic evolution")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Iteration")

        # Plot the average loss over time
        meanLoss = []
        percentile0 = []
        percentile25 = []
        percentile75 = []
        percentile100 = []
        for i in range(len(NEATLossData[0])):
            meanLoss.append(np.mean(NEATLossData[:, i]))
            percentile0.append(np.percentile(NEATLossData[:, i], 0))
            percentile25.append(np.percentile(NEATLossData[:, i], 25))
            percentile75.append(np.percentile(NEATLossData[:, i], 75))
            percentile100.append(np.percentile(NEATLossData[:, i], 100))
        ax[1].plot(np.arange(len(NEATLossData[0])), meanLoss, label="mean")
        ax[1].plot(np.arange(len(NEATLossData[0])), percentile0, color="red", label="0th and 100th percentile")
        ax[1].plot(np.arange(len(NEATLossData[0])), percentile25, color="black", label="25th and 75 percentile")
        ax[1].plot(np.arange(len(NEATLossData[0])), percentile75, color="black")
        ax[1].plot(np.arange(len(NEATLossData[0])), percentile100, color="red")
        ax[1].set_title("NEAT")
        ax[1].set_xlabel("Iteration")

        plt.legend()
        plt.show()

    @staticmethod
    def BaseLineVsNEATStructuralDistance():
        path = "Data\\NeatTest\\me\\False\\1"
        _, baseDistances, _, _, _ = Analysis.retrieveDataFromCsv(path)
        baseDistances = np.array(baseDistances)
        baseDistances = np.repeat(baseDistances, 4, axis=1)
        baseDistances = baseDistances[:, 0:10]

        path = "Data\\NeatTest\\me\\True\\1"
        _, NEATDistances, _, _, _ = Analysis.retrieveDataFromCsv(path)
        NEATDistances = np.array(NEATDistances)
        NEATDistances = np.repeat(NEATDistances, 4, axis=1)
        NEATDistances = NEATDistances[:, 0:10]

        # Remove models that have a distance measure below 0.
        baseDistances = baseDistances[baseDistances[:, 0] >= 0]
        NEATDistances = NEATDistances[NEATDistances[:, 0] >= 0]

        fig, ax = plt.subplots(1,2, sharey="all")
        fig.set_size_inches(18.5/2, 10.5/2)

        # Plot the average loss over time
        meanDistance = []
        percentile0 = []
        percentile25 = []
        percentile75 = []
        percentile100 = []
        for i in range(len(baseDistances[0])):
            meanDistance.append(np.mean(baseDistances[:, i]))
            percentile0.append(np.percentile(baseDistances[:, i], 0))
            percentile25.append(np.percentile(baseDistances[:, i], 25))
            percentile75.append(np.percentile(baseDistances[:, i], 75))
            percentile100.append(np.percentile(baseDistances[:, i], 100))
        ax[0].plot(np.arange(len(baseDistances[0])), meanDistance, label="mean")
        ax[0].plot(np.arange(len(baseDistances[0])), percentile0, color="red", label="0th and 100th percentile")
        ax[0].plot(np.arange(len(baseDistances[0])), percentile25, color="black", label="25th and 75th percentile")
        ax[0].plot(np.arange(len(baseDistances[0])), percentile75, color="black")
        ax[0].plot(np.arange(len(baseDistances[0])), percentile100, color="red")
        ax[0].set_title("Basic evolution")
        ax[0].set_ylabel("Structural Distance")
        ax[0].set_xlabel("Iteration")

        # Plot the average loss over time
        meanDistance = []
        percentile0 = []
        percentile25 = []
        percentile75 = []
        percentile100 = []
        for i in range(len(NEATDistances[0])):
            meanDistance.append(np.mean(NEATDistances[:, i]))
            percentile0.append(np.percentile(NEATDistances[:, i], 0))
            percentile25.append(np.percentile(NEATDistances[:, i], 25))
            percentile75.append(np.percentile(NEATDistances[:, i], 75))
            percentile100.append(np.percentile(NEATDistances[:, i], 100))
        ax[1].plot(np.arange(len(NEATDistances[0])), meanDistance, label="mean")
        ax[1].plot(np.arange(len(NEATDistances[0])), percentile0, color="red", label="0th and 100th percentile")
        ax[1].plot(np.arange(len(NEATDistances[0])), percentile25, color="black", label="25th and 75th percentile")
        ax[1].plot(np.arange(len(NEATDistances[0])), percentile75, color="black")
        ax[1].plot(np.arange(len(NEATDistances[0])), percentile100, color="red")
        ax[1].set_title("NEAT")
        ax[1].set_xlabel("Iteration")

        plt.legend()
        plt.show()

    @staticmethod
    def LossvsDistanceTest(genomeSampleSize=10000, samples=1000,
                           distributionSampleSize=1000, detail=20,
                           optimizerIters=40, maxPopSize=100, batchSize=10,
                           dataIndex=0, genomes=None):
        '''
        Calculates loss vs structural distance.
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
        rootPath = "Data\\KLossvsDistance\\" + str(dataIndex)
        while exists(rootPath):
            dataIndex += 1
            rootPath = "Data\\LossvsDistance\\" + str(dataIndex)
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
    def plotLossvsDistanceData():
        path = "Data\\LossvsDistance\\0"
        allPairs, allDistances, allFinalLosses, allLossData, allKLDivs = [], [], [], [], []

        pairs, distances, divergences, lossData, KLDivs = Analysis.retrieveDataFromCsv(path)
        allPairs.extend(pairs)
        allDistances.extend(distances)
        allFinalLosses.extend(divergences)
        allLossData.extend(lossData)
        allKLDivs.extend(KLDivs)
        
        # For both all losses invert the sign. Our optimizers maximize fitnesses, so we gave it the negative loss.
        # To properly plot the loss, we have to revert it back.
        for i in range(len(allFinalLosses)):
            allFinalLosses[i] *= -1

        for i in range(len(allLossData)):
            allLossData[i] = np.array(allLossData[i]) * -1

        weights = np.arange(0, 40)
        allLossData = np.array(allLossData)
        allDistances = np.array(allDistances)
        allLossData2 = np.sum(allLossData * weights, axis=1)
        indices = np.argsort(allLossData2)

        allLossData = allLossData.take(indices, 0)
        allDistances = allDistances.take(indices, 0)

        # Plot loss data ordered by structural distance
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(18.5, 10.5)

        for x, (listo, dist) in enumerate(zip(allLossData, allDistances)):
            ax.plot3D(np.ones(40)*dist, np.arange(0, 40), np.array(listo),
                             linewidth=2, antialiased=False)
        ax.set_xlabel("Structural distance")
        ax.set_ylabel("Iteration")
        ax.set_zlabel("Loss")
        plt.show()

        # Plot loss data ordered by weighted loss. Creates a nice surface to showcase the frequency of loss trajectories
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig.set_size_inches(18.5, 10.5)

        for x, (listo, dist) in enumerate(zip(allLossData, allDistances)):
            ax.plot3D(np.ones(40)*x, np.arange(0, 40), np.array(listo),
                      linewidth=2, antialiased=False)
        ax.set_xlabel("sorted by area under the curve")
        ax.set_ylabel("Iteration")
        ax.set_zlabel("Loss")
        plt.show()

        # Plot the final loss vs structural distance
        plt.scatter(x=allDistances, y=allFinalLosses, label="outcomes")
        xMin, xMax = min(allDistances), max(allDistances)
        yMin, yMax = min(allFinalLosses), max(allFinalLosses)

        # Plot a linear regression curve over the data
        allFinalLosses = np.array(allFinalLosses)
        regX = []
        regY = []
        for (x, y) in zip(allDistances, allFinalLosses):
            regX.append([x])
            regY.append([y])
        # Apply linear regression with structural distance as predictor
        reg = LinearRegression().fit(regX, regY)

        # Plot a line
        x = np.array([xMin, xMax])
        y = np.array([reg.coef_ * xMin + reg.intercept_, reg.coef_ * xMax + reg.intercept_]).flatten()
        plt.plot(x, y, color="red", label="linear regression")
        plt.legend()

        plt.xticks([xMin, xMax])
        plt.yticks([yMin, yMax])
        plt.xlabel("Structural distance")
        plt.ylabel("Loss")
        plt.show()

    @staticmethod
    def WithVsWithoutStructuralChange():
        path = "Data\\LossvsDistance\\0"
        _, _, _, withoutChangeLossData, _ = Analysis.retrieveDataFromCsv(path)
        withoutChangeLossData = np.array(withoutChangeLossData)

        path = "Data\\NeatTest\\me\\False\\0"
        _, _, _, withChangeLossData, _ = Analysis.retrieveDataFromCsv(path)
        withChangeLossData = np.array(withChangeLossData)

        # Invert losses. We converted loss to fitness, as both optimizers maximize fitness. Now we revert back to loss.
        withoutChangeLossData *= -1
        withChangeLossData *= -1

        # Remove models that started out already perfectly matching
        withoutChangeLossData = withoutChangeLossData[withoutChangeLossData[:, 0] != 0]
        withChangeLossData = withChangeLossData[withChangeLossData[:, 0] != 0]

        fig, ax = plt.subplots(1,2, sharey="all")
        fig.set_size_inches(18.5/2, 10.5/2)

        # Plot the average loss over time
        meanLoss = []
        percentile0 = []
        percentile25 = []
        percentile75 = []
        percentile100 = []
        for i in range(len(withoutChangeLossData[0])):
            meanLoss.append(np.mean(withoutChangeLossData[:, i]))
            percentile0.append(np.percentile(withoutChangeLossData[:, i], 0))
            percentile25.append(np.percentile(withoutChangeLossData[:, i], 25))
            percentile75.append(np.percentile(withoutChangeLossData[:, i], 75))
            percentile100.append(np.percentile(withoutChangeLossData[:, i], 100))
        ax[0].plot(np.arange(len(withoutChangeLossData[0])), meanLoss, label="mean")
        ax[0].plot(np.arange(len(withoutChangeLossData[0])), percentile0, color="red", label="0th and 100th percentile")
        ax[0].plot(np.arange(len(withoutChangeLossData[0])), percentile25, color="black", label="25th and 75th percentile")
        ax[0].plot(np.arange(len(withoutChangeLossData[0])), percentile75, color="black")
        ax[0].plot(np.arange(len(withoutChangeLossData[0])), percentile100, color="red")
        ax[0].set_title("Without structural change")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Iteration")

        # Plot the average loss over time
        meanLoss = []
        percentile0 = []
        percentile25 = []
        percentile75 = []
        percentile100 = []
        for i in range(len(withChangeLossData[0])):
            meanLoss.append(np.mean(withChangeLossData[:, i]))
            percentile0.append(np.percentile(withChangeLossData[:, i], 0))
            percentile25.append(np.percentile(withChangeLossData[:, i], 25))
            percentile75.append(np.percentile(withChangeLossData[:, i], 75))
            percentile100.append(np.percentile(withChangeLossData[:, i], 100))
        ax[1].plot(np.arange(len(withChangeLossData[0])), meanLoss, label="mean")
        ax[1].plot(np.arange(len(withChangeLossData[0])), percentile0, color="red", label="0th and 100th percentile")
        ax[1].plot(np.arange(len(withChangeLossData[0])), percentile25, color="black", label="25th and 75 percentile")
        ax[1].plot(np.arange(len(withChangeLossData[0])), percentile75, color="black")
        ax[1].plot(np.arange(len(withChangeLossData[0])), percentile100, color="red")
        ax[1].set_title("With structural change")
        ax[1].set_xlabel("Iteration")

        plt.legend()
        plt.show()

        plt.hist(withoutChangeLossData[:, len(withoutChangeLossData[0])-1],
                 edgecolor='black', linewidth=1.2, bins=np.arange(-0.05,1.05,0.025).tolist(), label="Without change")
        plt.hist(withChangeLossData[:, len(withChangeLossData[0])-1], color="green",
                 edgecolor='black', linewidth=1.2, bins=np.arange(-0.05,1.05,0.025).tolist(), label="With change")
        plt.title("With vs without structural change")
        plt.xlabel("Final loss")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    @staticmethod
    def distanceTest(genomes, pair):
        genome = genomes[pair[0]]
        otherGenome = genomes[pair[1]]
        return DistanceMetric.run(genome, otherGenome)

    @staticmethod
    def KLDivTest(genomes, pair, sampleSize=1000, detail=20, iterations=40, maxPopSize=200, batchSize=100):
        env = VariableEnv(genomes[pair[0]], datapointCount=sampleSize, detail=detail)
        otherGenome = genomes[pair[1]]

        optimizer = ProbabilisticNEAT(iterations,
                                      maxPopSize=maxPopSize, batchSize=batchSize,
                                      weightsOnly=True, useMerging=False, useSpeciation=False)

        otherGenome, losses, bestGenes = optimizer.run(otherGenome, env)

        KLDiv = env.discretizedKullbackLeibler(env.generated, otherGenome.generate(env.input))

        return otherGenome.fitness, losses, KLDiv

    @staticmethod
    def NeatTest(genomes, index, sampleSize=1000, detail=20, iterations=40, maxPopSize=200, batchSize=100,
                 useMergingAndSpeciation=False, criterion="me"):
        env = VariableEnv(genomes[index], datapointCount=sampleSize, detail=detail, criterion=criterion)
        otherGenome = ProbabilisticGenome(env.inputs(), env.outputs())

        optimizer = ProbabilisticNEAT(iterations,
                                      maxPopSize=maxPopSize, batchSize=batchSize,
                                      weightsOnly=False,
                                      useMerging=useMergingAndSpeciation, useSpeciation=useMergingAndSpeciation)

        otherGenome, losses, bestGenomes = optimizer.run(otherGenome, env)
        distances = []
        for i, genome in enumerate(bestGenomes):
            if i % 4 == 0:
                distances.append(DistanceMetric.run(env.model, genome, maxTime=20))

        KLDiv = env.discretizedKullbackLeibler(env.generated, otherGenome.generate(env.input))

        return otherGenome.fitness, losses, KLDiv, distances

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
    def storeDataInCsv(pairs, distances, finalLosses, lossData, KLDivs, rootPath="Data"):
        path = rootPath + "\\Data.csv"

        fieldnames = ["Env", "Model", "Structural Distance", "finalLoss", "Losses", "KLDiv"]
        with open(path, mode='w') as file:
            data_writer = csv.DictWriter(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                         fieldnames=fieldnames)
            for (i, compareTo), distance, finalLoss, losses, KLDiv in \
                    zip(pairs, distances, finalLosses, lossData, KLDivs):
                data_writer.writerow({"Env": i,
                                      "Model": compareTo,
                                      "Structural Distance": distance,
                                      "finalLoss": finalLoss,
                                      "Losses": losses,
                                      "KLDiv": KLDiv})

    @staticmethod
    def retrieveDataFromCsv(rootPath="Data"):
        path = rootPath + "\\Data.csv"
        pairs = []
        distances = []
        finalLoss = []
        lossData = []
        KLDivs = []
        fieldnames = ["Env", "Model", "Structural Distance", "finalLoss", "Losses", "KLDiv"]
        if exists(path):
            with open(path, mode='r') as file:
                data_reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL,
                                             fieldnames=fieldnames)
                for row in tqdm(data_reader, desc= "Loading Data from CSV", unit= "rows"):
                    pairs.append([ast.literal_eval(row["Env"]), ast.literal_eval(row["Model"])])
                    distances.append(ast.literal_eval(row["Structural Distance"]))
                    finalLoss.append(ast.literal_eval(row["finalLoss"]))
                    lossData.append(ast.literal_eval(row["Losses"]))
                    if row["KLDiv"] is not None:
                        KLDivs.append(ast.literal_eval(row["KLDiv"]))
        return pairs, distances, finalLoss, lossData, KLDivs



