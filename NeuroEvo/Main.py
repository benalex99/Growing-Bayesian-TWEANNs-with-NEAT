import random
import time

import matplotlib.pyplot as plt
import numpy as np
from NeuroEvo.Environments.GymEnv import GymEnv
from NeuroEvo.Optimizers.NEAT.NEAT import NEAT
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.Optimizers.QLearner.QLearner import QPolicy
from NeuroEvo.Optimizers.Trainer import Trainer
import NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPEnvironment as Env
from pyro.distributions import *
import torch
from NeuroEvo.NeuralNetwork.EnsembleNN.DiscreteWeightBNN import DWBNN
from NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPCategoricalAgent import DP as DPC
from NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPExample import DP as DPE
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticNEAT import ProbabilisticNEAT


def neatTest():
    optim = NEAT(iterations=1000000000000, maxPopSize=10, batchSize=20, episodeDur=400, showProgress=(1, 10))
    # env = GymEnv('CartPole-v0')
    # env = GymEnv('MountainCar-v0')
    env = GymEnv('LunarLander-v2')
    # # env = GymEnv('LunarLanderContinuous-v2')
    # # env = GymEnv('Acrobot-v1')
    # # env = GymEnv('Pendulum-v0')
    # # env = GymEnv('MountainCarContinuous-v0')
    # # env = GymEnv('BipedalWalker-v3')
    # # env = GymEnv('Copy-v0')
    #
    # print(env.outputs())
    gg, score = Trainer.run(optim, env)
    gg.visualize(ion=False)


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


def speciationTest():
    avgSpecies = 0
    iter = 500
    for x in range(iter):
        genomes = []
        genome = NEATGenome(5, 1)
        genome.fitness = random.randint(-10, 10)
        genomes.append(genome)
        optim = NEAT(iterations=1000000000000, batchSize=200, maxPopSize=100, episodeDur=400, showProgress=(1, 1000))

        for i in range(50):
            genomeN = genomes[random.randint(0, len(genomes) - 1)].copy()
            genomeN.mutate(i)
            genomeN.fitness = random.randint(-50, 50)
            genomes.append(genomeN)

        optim.speciation(genomes, 1, 3, 5, 5)
        print(len(optim.species))
        avgSpecies += len(optim.species)
    print("Average number of Species:" + str(avgSpecies / iter))

def generativeModelTest():

    genome = ProbabilisticGenome(5, 2)
    for i in range(10):
        genome.mutate(i)
    print(genome.nodeStats())
    genome.visualize()
    plt.pause(20)
    # genome.updateLayers()
    # print(genome)

    data = genome.generate(np.ones((5,5)))
    print(data)

def probNeatTest():
    optim = ProbabilisticNEAT(iterations=1000000000000, maxPopSize=10, batchSize=20, episodeDur=400, showProgress=(1, 100))
    env = GymEnv('LunarLander-v2')
    genome = ProbabilisticGenome(env.inputs(), env.outputs())
    optim.run(genome, env)


# nnToGenome()
# Testing.test()
# speciationTest()
# neatTest()
# DPC.test()

generativeModelTest()
# probNeatTest()