from Optimizers.NEAT.NEAT import NEAT
from Optimizers.Trainer import Trainer
from Environments.GymEnv import GymEnv
from Optimizers.QLearner.QLearner import QPolicy
from Genome.Genome import Genome
from Optimizers.NEAT.NEATGenome import NEATGenome
from Testing import Testing
import time
from NeuralNetwork.AbsoluteGrad.Linear import AbsGradTest
import gym
from Environments.Classification import BayesianClassification
import torch
from NeuroEvo.NeuralNetwork.EnsembleNN.DiscreteWeightBNN import DWBNN
import random


def neatTest():
    optim = NEAT(iterations=1000000000000, batchSize=200, maxPopSize=100, episodeDur=400, showProgress=(1, 1000))
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
    for x in range(20):
        genomes = []
        genome = NEATGenome(5, 1)
        genomes.append(genome)
        optim = NEAT(iterations=1000000000000, batchSize=200, maxPopSize=100, episodeDur=400, showProgress=(1, 1000))

        for i in range(20):
            genomeN = genomes[random.randint(0, len(genomes) - 1)].copy()
            genomeN.mutate(i)
            genomes.append(genomeN)

        optim.speciation(genomes, 0.5, 0.5, 1)
        print(len(optim.species))


# nnToGenome()
# Testing.test()
# speciationTest()
neatTest()