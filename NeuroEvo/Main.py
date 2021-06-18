import time

import numpy as np

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT import UnitTests
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.Analysis import Analysis
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.Optimizers.QLearner.QLearner import QPolicy
from NeuroEvo.NeuralNetwork.EnsembleNN.DiscreteWeightBNN import DWBNN


def Qlearning():
    qLearning = QPolicy('LunarLander-v2')
    qLearning.run(100000)

def BayesStuff():
    dwbnn = DWBNN(layers=[(1, 2)], weightCount=5)
    for _ in range(10):
        print(dwbnn([0]))

def nnToGenome():
    genome = NEATGenome(5, 1)
    for i in range(10):
        genome.mutate(i)
    genome.visualize()
    nn = genome.toNN()
    genome.weightsFromNN(nn)
    time.sleep(1000)

# UnitTests.neatTest()
# UnitTests.meLossTest()

# genomes = Analysis.readGenomesFromCsv()

# This runs the tests for the first research question
# Analysis.LossvsDistanceTest(genomes=genomes)
# Analysis.NeatVsNormalTest(genomes=genomes, useMergingAndSpeciation=False, criterion="me")

# This plots the results for the first research question
# Analysis.WithVsWithoutStructuralChange()
# Analysis.plotLossvsDistanceData()
# Analysis.plotNeatTestData(withSpeciationAndMerging=True, dataset=1)

# This runs the tests for the second research question
# Analysis.NeatVsNormalTest(genomes=genomes, useMergingAndSpeciation=False, criterion="me",
#                           samples=200, optimizerIters=10, batchSize=100, maxPopSize=50)
# Analysis.NeatVsNormalTest(genomes=genomes, useMergingAndSpeciation=True, criterion="me",
#                           samples=200, optimizerIters=10, batchSize=100, maxPopSize=50)

# This plots the results for the second research question
#Analysis.BaseLineVsNEATLoss()
#Analysis.BaseLineVsNEATStructuralDistance()

