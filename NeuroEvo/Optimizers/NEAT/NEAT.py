import random

from NeuroEvo.Genome.NodeGene import NodeGene
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
import numpy as np
import time


class NEAT:
    population: list

    def __init__(self, iterations, batchSize, maxPopSize, episodeDur, showProgress=(0, 0)):
        self.population = []
        self.iterations = iterations
        self.batchSize = batchSize
        self.maxPopSize = maxPopSize
        self.episodeDur = episodeDur
        self.showProgress = showProgress
        return

    def run(self, rootGenome, env, seed=0):
        self.hMarker = 1
        self.population.append(rootGenome)
        self.visualize(rootGenome, env, 500, useDone=False)
        for iter in range(self.iterations):
            print("Iteration: " + str(iter))
            ntime = time.time()
            avgSize = 0
            toBeTested = []

            # Create a bunch of mutations
            for i in range(self.batchSize):
                g = self.newGenome()
                avgSize += len(g.edges)
                toBeTested.append(g)

            print("AvgSize : " + str(avgSize / len(toBeTested)))

            # Assign values to the mutations and add them to the population
            env.test(toBeTested, self.episodeDur, seed=iter)
            self.population = self.population + toBeTested
            self.population.sort(key=lambda x: x.fitness, reverse=True)

            # Discard bad mutations from the population until the max population count is reached
            while (len(self.population) > self.maxPopSize):
                i = random.randint(max(0, int(len(self.population) / 2 - 1)), len(self.population) - 1)
                self.population.remove(self.population[i])
            self.median = self.population[int(len(self.population) / 2)].fitness
            print("Took : " + str(time.time() - ntime))
            print("Median score: " + str(self.median))
            print("Best Score: " + str(self.bestGene().fitness) + "\n")
            if (self.showProgress[0] > 0 and iter % self.showProgress[0] == 0):
                self.visualize(self.bestGene(), env, self.showProgress[1], seed=iter)

        # Get the best gene from the population
        return self.bestGene()
        # if(self.population)

    def merge(self, firstGenome: NEATGenome, secondGenome: NEATGenome):
        fittestGenome: NEATGenome = firstGenome.copy() if firstGenome.fitness >= secondGenome.fitness else secondGenome.copy()
        weakestGenome: NEATGenome = firstGenome.copy() if firstGenome.fitness < secondGenome.fitness else secondGenome.copy()

        for index, value in enumerate(weakestGenome.edges):
            if index < len(fittestGenome.edges) and fittestGenome.edges[
                min(index, len(fittestGenome.edges) - 1)].hMarker == value.hMarker:
                if random.randint(0, 1) < 1:
                    fittestGenome.edges[index] = value
            elif firstGenome.fitness == secondGenome.fitness:
                while (len(fittestGenome.nodes) <= value.toNr or len(fittestGenome.nodes) <= value.fromNr):
                    fittestGenome.nodes.append(NodeGene(nodeNr=len(fittestGenome.nodes)))
                if (fittestGenome.nodes[value.fromNr].layer <= fittestGenome.nodes[value.toNr].layer and (not fittestGenome.nodes[value.fromNr].outputtingTo.__contains__(value.toNr))):
                    fittestGenome.edges.append(value)
                    fittestGenome.nodes[value.fromNr].outputtingTo.append(value.toNr)
                    fittestGenome.increaseLayers(fittestGenome.nodes[value.fromNr], fittestGenome.nodes[value.toNr])

        return fittestGenome

    def bestGene(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        bestGene = self.population[0]
        for genome in self.population:
            if (genome.fitness > bestGene.fitness):
                bestGene = genome
        return bestGene

    def visualize(self, gene, env, duration, useDone=True, seed=0):
        gene.visualize()
        env.visualize(gene, duration=duration, useDone=useDone, seed=seed)

    def newGenome(self):
        if (random.randint(0, 1) < 1 or len(self.population) <= 3):
            g = self.population[random.randint(0, max(0, int(len(self.population) / 2 - 1)))].copy()
            self.hMarker = self.hMarker + g.mutate(self.hMarker)
        else:
            g1 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            g2 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            while (g1 == g2):
                g2 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            g = self.merge(self.population[g1],
                           self.population[g2])
        return g
