from Optimizers.NEAT.NEAT import NEAT
from Optimizers.Trainer import Trainer
from Environments.GymEnv import GymEnv
from Optimizers.QLearner.QLearner import QPolicy
from Genome.Genome import Genome
from Optimizers.NEAT.NEATGenome import NEATGenome
import matplotlib.pyplot as plt
import numpy as np
from Testing import Testing
import time
from NeuralNetwork.AbsoluteGrad.Linear import AbsGradTest
import gym
from Environments.Classification import BayesianClassification
import torch
from NeuroEvo.NeuralNetwork.EnsembleNN.DiscreteWeightBNN import DWBNN
from NeuralNetwork.HierarchicalDirichletProcess.DP import DP

def neatTest():
    optim = NEAT(iterations= 1000000000000, batchSize= 200, maxPopSize= 100, episodeDur= 400, showProgress= (1,1000))
    env = GymEnv('CartPole-v0')
    # env = GymEnv('MountainCar-v0')
    # env = GymEnv('LunarLander-v2')
    # # env = GymEnv('LunarLanderContinuous-v2')
    # # env = GymEnv('Acrobot-v1')
    # # env = GymEnv('Pendulum-v0')
    # # env = GymEnv('MountainCarContinuous-v0')
    # # env = GymEnv('BipedalWalker-v3')
    # # env = GymEnv('Copy-v0')
    #
    # print(env.outputs())
    gg, score = Trainer.run(optim,env)
    gg.visualize(ion= False)

def Qlearning():
    qLearning = QPolicy('LunarLander-v2')
    qLearning.run(100000)

def BayesStuff():
    dwbnn = DWBNN(layers= [(1, 2)], weightCount= 5)
    for _ in range(10):
        print(dwbnn([0]))

def nnToGenome():
    genome = NEATGenome(5,1)
    for i in range(50):
        genome.mutate(i)
    genome.visualize()
    genome.increaseInput()
    time.sleep(3)
    genome.visualize()
    nn = genome.toNN()
    nn.fromToLayers[0][0].weight[0] *= 2
    genome.weightsFromNN(nn)
    nn2 = genome.toNN()
    print((nn.fromToLayers[0][0].weight == nn2.fromToLayers[0][0].weight).all())
    print(nn.fromToLayers[0][0].weight)

    plt.pause(1000)


#nnToGenome()
#Testing.test()

# print(pred-data)
dp = DP()
dp.test()
