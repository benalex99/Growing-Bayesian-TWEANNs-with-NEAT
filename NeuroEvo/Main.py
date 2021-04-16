import random
import time

from NeuroEvo.Environments.GymEnv import GymEnv
from NeuroEvo.Optimizers.NEAT.NEAT import NEAT
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.Optimizers.QLearner.QLearner import QPolicy
from NeuroEvo.Optimizers.Trainer import Trainer
from NeuroEvo.NeuralNetwork.UnclassifiedExperiments.chess_questionmark_.myChess import myChess
import NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPEnvironment as Env
from  pyro.distributions import *
import torch
from NeuroEvo.NeuralNetwork.EnsembleNN.DiscreteWeightBNN import DWBNN
from NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPCategoricalAgent import DP as DPC
from NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPExample import DP as DPE


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
    avgSpecies = 0
    iter = 10
    for x in range(iter):
        genomes = []
        genome = NEATGenome(5, 1)
        genomes.append(genome)
        optim = NEAT(iterations=1000000000000, batchSize=200, maxPopSize=100, episodeDur=400, showProgress=(1, 1000))

        for i in range(20000):
            genomeN = genomes[random.randint(0, len(genomes) - 1)].copy()
            genomeN.mutate(i)
            genomes.append(genomeN)

        optim.speciation(genomes, 0.5, 0.5, 1, 8)
        print(len(optim.species))
        avgSpecies += len(optim.species)
    print("Average number of Species:" + str(avgSpecies/iter))


# nnToGenome()
# Testing.test()
# speciationTest()
# neatTest()
# DPC.test()

# print(Categorical(torch.Tensor([1])).sample([1]))
yo = myChess()