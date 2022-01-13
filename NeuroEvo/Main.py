import time

import numpy as np

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT import UnitTests
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.Analysis import Analysis
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.Optimizers.QLearner.QLearner import QPolicy


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

UnitTests.neatTest()
# UnitTests.meLossTest()

# genomes = Analysis.readGenomesFromCsv()


