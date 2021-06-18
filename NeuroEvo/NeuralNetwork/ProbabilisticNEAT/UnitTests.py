import random
import torch

from matplotlib import pyplot as plt
from pyro.infer import TraceGraph_ELBO

from NeuroEvo.Environments.GymEnv import GymEnv
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT import NEATEnv, StructuralDistance
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.Analysis import Analysis
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.NEATEnv import VariableEnv
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticNEAT import ProbabilisticNEAT
from NeuroEvo.Optimizers.NEAT.NEAT import NEAT
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
from NeuroEvo.Optimizers.Trainer import Trainer

# Testing optimization of our models with SVI. Does partially work.
# Will diverge from optimal solutions, hence discarded.
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

# Testing optimization of our models with MCMC. Does not work. Cannot find legal starting parameters. Discarded.
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
# The outcome showed a predictable distribution of genomes. Later NEAT test verifies. Works well.
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

# Testing our implementation of NEAT on openAI gym "Lunar Lander". Works well.
# Achieves a score of 180 (solved is 200) after 100 iterations with the current parameter settings. Took about 10 minutes
def neatTest():
    env = GymEnv('LunarLander-v2')
    rootGenome = ProbabilisticGenome(env.inputs(), env.outputs())
    optim = ProbabilisticNEAT(iterations=100, maxPopSize=100, batchSize=200, episodeDur=400, showProgress=(1, 200),
                 inclusionThreshold=5, useMerging=True, useSpeciation=True, useDistributions=False)
    optimized, losses, bestGenes = optim.run(rootGenome, env, showProgressBar=True)
    optimized.visualize()

# Testing arbitrary model generation, as well as their ability to generate data. Works well.
def generativeModelTest():
    genome = ProbabilisticGenome(2, 2)
    for i in range(10):
        genome.mutate(i)
    print(genome.nodeStats())
    genome.visualize()

    data = genome.generate(torch.ones((50, 2), device=torch.device('cuda')))
    print(data)
    print(genome)

    plt.pause(10000)

# Testing our random weight optimization using NEAT without merging or speciation. Works.
def RandomOptim():
    envGenome = ProbabilisticGenome(1,1)
    for graphs in range(10):
        i = 0
        while i < 15:
            i += envGenome.mutate(i)
        env = VariableEnv(envGenome, datapointCount=1000)

        model = envGenome.copy()
        for i in range(15):
            model.tweakWeight()

        # genome = ProbabilisticGenome(env.inputs(), env.outputs())

        optim = ProbabilisticNEAT(iterations=40, maxPopSize=100, batchSize=10,
                                  useMerging=False, useSpeciation=False, weightsOnly=True)
        best, loss = optim.run(model, env, debug=False, showProgressBar=True)

        plt.plot(range(len(loss)), loss)
        print(best.fitness)
    plt.show()
    plt.pause(1000)

# Testing our structure comparison metric. Kinda sufficient.
# May take a long time for large networks (beyond the sizes that we test)
def structuralDistanceTest():
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

# Testing our distribution comparison metric. Works well.
def meLossTest():
    genome = ProbabilisticGenome(1, 1)
    for i in range(10):
        genome.mutate(i)
    genome2 = genome.copy()
    env = NEATEnv.VariableEnv(genome, datapointCount=1000)
    print(env.testOne(genome2))
    genome.visualize()
    plt.pause(5)
    for i in range(10, 20, 1):
        genome2.mutate(i)
    print(env.testOne(genome2))
    genome2.visualize()
    plt.pause(10000)

# Testing storing and retrieving Genomes on and from the hard disk. Works.
def genomeStorageTest():
    genome = ProbabilisticGenome(1, 1)
    for i in range(10):
        genome.mutate(i)
    genome.visualize()
    plt.pause(3)
    print("next")
    Analysis.storeGenomesInCsv([genome])
    [genome2] = Analysis.readGenomesFromCsv()
    genome2.visualize()
    plt.pause(3)
    for i in range(10):
        genome2.mutate(i)
    genome2.visualize()
    plt.pause(3)
    genome.generate()

# Testing the new structural distance metric that performs greedy tree search on the mapping space. Works well.
def structuralDistanceTest2():
    genome = ProbabilisticGenome(1, 1)
    for i in range(10):
        genome.mutate(i)
    genome2 = genome.copy()
    genome.visualize()
    print(StructuralDistance.DistanceMetric.run(genome, genome2, 20000, typeMatters=True))
    plt.pause(3)
    successes = 0
    for i in range(10):
        success = 0
        while (success<=0):
            success = genome2.mutate(successes)
            successes += success
    genome2.visualize()
    print(StructuralDistance.DistanceMetric.run(genome, genome2, 20000, typeMatters=True))
    plt.pause(10000)