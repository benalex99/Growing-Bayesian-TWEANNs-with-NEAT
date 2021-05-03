import random
import time

import matplotlib.pyplot as plt
import numpy as np
import math
from pyro.infer import TraceGraph_ELBO
from tqdm import tqdm

from NeuroEvo.Environments.GymEnv import GymEnv
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT import NEATEnv
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.AdvancedEdgeGene import AdvancedEdgeGene
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.AdvancedNodeGene import AdvancedNodeGene
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.Analysis import Analysis
from NeuroEvo.Optimizers.NEAT.NEAT import NEAT
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.Optimizers.QLearner.QLearner import QPolicy
from NeuroEvo.Optimizers.Trainer import Trainer
from pyro.distributions import *
import NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPEnvironment as Env
import torch
from NeuroEvo.NeuralNetwork.EnsembleNN.DiscreteWeightBNN import DWBNN
from NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPCategoricalAgent import DP as DPC
from NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPExampleUnivariate import DP as DPE
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticNEAT import ProbabilisticNEAT
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.CustomDistr import ReluDistr

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

def sviTest():
    torch.cuda.init()
    inputs = 2
    outputs = 1
    datapoints = 10000
    env = VariableEnv(inputs=inputs, outputs=outputs, mutations=20, datapointCount=datapoints)
    # genome = ProbabilisticGenome(env.inputs(), env.outputs())
    genome = env.model.copy()

    plt.figure(0)
    env.model.visualize()
    # plt.pause(3)

    # mutations = 0
    # for _ in range(10):
    #     mutations += genome.mutate(mutations)
    mutations = 0
    for _ in range(0):
        genome.tweakWeight()

    plt.figure(1)
    genome.visualize()
    # plt.pause(3)

    losses, lastLoss = genome.train(env.data, 1000, loss=TraceGraph_ELBO())

    plt.figure(2)
    genome.visualize()
    plt.pause(3)

    plt.figure(3)
    plt.subplot(1,1,1)
    plt.plot(losses)

    plt.figure(4)
    data = env.generated
    for i in range(outputs):
        plt.subplot(2, outputs, i+1)
        plt.title("Env output " + str(i))
        plt.hist(list(np.array(data[:,i].tolist()).flat), bins = 100, density=True)

    input = torch.ones((datapoints, inputs), device=torch.device('cuda'))
    data = genome.generate(input)
    for i in range(outputs):
        plt.subplot(2, outputs, outputs + i+1)
        plt.title("Model output " + str(i))
        plt.hist(list(np.array(data[:,i].tolist()).flat), bins = 100, density=True)
    plt.tight_layout()
    plt.show()
    plt.pause(1000)

def MCMCTest():
    from pyro.infer import MCMC, NUTS

    inputs = 1
    outputs = 1
    datapoints = 10000
    env = VariableEnv(inputs=inputs, outputs=outputs, mutations=20, datapointCount=datapoints)
    # genome = ProbabilisticGenome(env.inputs(), env.outputs())
    genome = env.model.copy()
    nuts_kernel = NUTS(genome.model)
    # 1000 and 200
    mcmc = MCMC(nuts_kernel, num_samples=100, warmup_steps=50)

    mcmc.run(env.data)

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    print(hmc_samples)


# Old function to check the behavior of genomes being allocated to species.
# The outcome showed a predictable distribution of genomes
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

        optim.speciation()
        print(len(optim.species))
        avgSpecies += len(optim.species)
    print("Average number of Species:" + str(avgSpecies / iter))

# Testing our implementation of NEAT on openAI gym "Lunar Lander" and "CartPole"
def neatTest():
    optim = NEAT(iterations=1000000000000, maxPopSize=100, batchSize=200, episodeDur=400, showProgress=(1, 200),
                 inclusionThreshold=5)
    env = GymEnv('LunarLander-v2')
    # env = GymEnv('CartPole-v0')
    gg, score = Trainer.run(optim, env)
    gg.visualize(ion=False)

def generativeModelTest():

    genome = ProbabilisticGenome(2, 2)
    for i in range(10):
        genome.mutate(i)
    print(genome.nodeStats())
    genome.visualize()

    data = genome.generate(torch.ones((50, 2),device=torch.device('cuda')))
    print(data)
    print(genome)

    plt.pause(10000)

class VariableEnv:
    def __init__(self, inputs = 1, outputs = 1, mutations = 10, datapointCount = 1000):
        generativeModel = ProbabilisticGenome(inputs, outputs)
        for i in range(mutations):
            generativeModel.mutate(i)
        self.inputCount = inputs
        self.outputCount = outputs
        self.model = generativeModel
        self.input = torch.ones((datapointCount, inputs), device=torch.device('cuda'))
        self.generated = self.model.generate(self.input)
        self.data = [self.input, self.generated]

    def test(self, population, duration, seed):
        tests = []
        for genome in population:
            if genome.fitness == -math.inf:
                tests.append(genome)
        for genome in tqdm(tests):
            predictions = genome.generate(self.input)
            genome.fitness = -self.discretizedKullbackLeiber(self.generated, predictions, punishIgnorance=True)
            # genome.fitness = -self.mse_loss(predictions, self.generated)

    def mse_loss(self, predictions, targets):
        return torch.sum((predictions - targets) ** 2)

    @staticmethod
    def discretizedKullbackLeiber(targets, predictions, detail=20, punishIgnorance=False):
        """
        :param predictions: Samples of the model Q
        :param targets: Samples of the environment P
        :param detail: The detail at which we compare. Determines the amount of buckets. Can be any positive real.
        :param punishIgnorance: Whether to punish a disjoint probability space
                                    (Samples from P that have a probability of 0 according to Q)
        :return: The approximated Kullback-Leibler divergence KL(P||Q)
        """
        # We batch continuous variables together to form buckets. This is a form of clustering.
        # First we normalize the continuous data to a range of 100.
        # Then we scale the values up by the detail parameter
        # and round to the closest integer, which now represent class indices.
        # Once we have classes, we can apply KL-Divergence as would be on discrete distributions.

        # Normalize each dimension across all samples.
        max = torch.stack([predictions, targets]).max(0)[0][0]
        max[max==0] = 1 # Don't divide by zero :)
        predNormed = predictions / max
        targetsNormed = targets / max

        # Scale up and round to closest integers, so we have up to D * detail different buckets
        predNormed = torch.round(predNormed * detail)
        targetsNormed = torch.round(targetsNormed * detail)

        # Count the occurrences in each bucket
        uniquePreds = {}
        for pred in predNormed:
            key = str(pred.tolist())
            if uniquePreds.keys().__contains__(key):
                uniquePreds[key] += 1
            else:
                uniquePreds[key] = 1
        uniqueTargets = {}
        for target in targetsNormed:
            key = str(target.tolist())
            if uniqueTargets.keys().__contains__(key):
                uniqueTargets[key] += 1
            else:
                uniqueTargets[key] = 1

        # Calculate the probabilities of the values based on the occurrences
        q = {}
        for pred in uniquePreds.keys():
            q[pred] = uniquePreds[pred] / len(predictions)
        p = {}
        for target in uniqueTargets.keys():
            p[target] = uniqueTargets[target] / len(targets)

        # Calculate the Kullback-Leibler divergence KL(P||Q)
        klDivergence = 0
        for x in q.keys():
            if p.keys().__contains__(x):
                klDivergence += p[x] * math.log(p[x]/q[x])

        # TODO: What about values for P that are not supported by Q?
        #  If we do not punish this, the model will attempt to
        #  optimize itself for having a disjoint probability space
        # Suggestion: multiply final KL-divergence by 1+lostPProb. Worst case, KL divergence is doubled.
        if(punishIgnorance):
            lostPProb = 0
            for x in p.keys():
                if not q.keys().__contains__(x):
                    lostPProb += p[x]
            klDivergence *= 1+lostPProb
            # print("lostProb " + str(lostPProb))
            #                                                           __           __
            # If all predictions are disjoint, there is nothing we can do \ ( ^-^ ) /
            if lostPProb >= 1:
                klDivergence = 1000000 # A large value, because neat doesnt handle infinity well

        return klDivergence

    def inputs(self):
        return self.inputCount

    def outputs(self):
        return self.outputCount

    def visualize(self, gene, duration, useDone=None, seed=None):
        gene.visualize()
        plt.pause(duration/100)
        self.model.visualize()
        plt.pause(duration/100)

def RandomOptim():

    optim = ProbabilisticNEAT(iterations=1000000000000, maxPopSize=200, batchSize=200,
                              useMerging=False, useSpeciation=False, weightsOnly=True)
    #env = GymEnv('LunarLander-v2')
    env = VariableEnv(inputs=2, outputs=3, mutations=5, datapointCount=1000)
    # genome = ProbabilisticGenome(env.inputs(), env.outputs())
    genome = env.model.copy()
    for _ in range(1):
        genome.tweakWeight()
    optim.run(genome, env)

def compareStructureTest():
    genome = ProbabilisticGenome(1, 1)
    for i in range(10):
        genome.mutate(i)
    genome2 = genome.copy()
    print(Analysis.structuralDivergence(genome, genome2))
    successes = 0
    for i in range(5):
        success = 0
        while (success<=0):
            success = genome2.mutate(successes)
            successes += success
    print(successes)
    genome.visualize()
    print(Analysis.structuralDivergence(genome, genome2))
    plt.pause(10000)

def mseLossTest():
    genome = ProbabilisticGenome(1, 1)
    for i in range(10):
        genome.mutate(i)
    genome2 = genome.copy()
    env = NEATEnv.VariableEnv(genome, datapointCount=1000)
    print(env.testOne(genome2))
    genome.visualize()
    plt.pause(5)
    for i in range(10,20,1):
        genome2.mutate(i)
    print(env.testOne(genome2))
    genome2.visualize()
    plt.pause(10000)

# compareStructureTest()
# mseLossTest()
# generativeModelTest()
neatTest()

# print(Multinomial(1, logits=torch.tensor([1.,2.,1.])).sample([1]))
# print(Dirichlet(torch.tensor([1.,1.,1.])).sample([1]))