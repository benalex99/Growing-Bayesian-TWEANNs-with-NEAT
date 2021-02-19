import random
import numpy as np


class NEAT:
    population: list

    def __init__(self, env, rootGenome, iterations, batchSize, maxPopSize):
        self.population = []
        self.env = env
        self.hMarker = 1
        self.rootGenome = rootGenome
        self.iterations = iterations
        self.batchSize = batchSize
        self.maxPopSize = maxPopSize
        return

    def testRun(self):

        self.population.append(self.rootGenome)
        toBeTested = []
        for iter in range(self.iterations):
            print("Iteration: " + str(iter))
            # Create a bunch of mutations
            for i in range(self.batchSize):
                g = self.population[random.randint(0, len(self.population) - 1)].copy()
                self.hMarker = self.hMarker + g.mutate(self.hMarker)
                toBeTested.append(g)

            # Assign values to the mutations and add themto the population
            self.env.test(toBeTested)
            self.population = + toBeTested
            self.avgScore = 0
            for genome in self.population:
                self.avgScore += genome.fitness
            self.avgScore = self.avgScore / len(self.population)

            # Discard bad mutations from the population until the max population count is reached
            while (len(self.population) > self.maxPopSize):
                i = random.randint(0, len(self.population) - 1)
                if (self.population[i] < self.avgScore):
                    self.population.remove(i)

        # Get the best gene from the population
        bestGene = self.population[0]
        for genome in self.population:
            if (genome.fitness > bestGene.fitness):
                bestGene = genome

        return bestGene
        # if(self.population)

    def merge(self, stGenome, ndGenome):
        genomeA = stGenome.copy()
        genomeB = ndGenome.copy()
        fittestGenome = lambda x: genomeA if genomeA.score > genomeB.score else genomeB
        hMarkerMaxA = 0
        hMarkerMaxB = 0

        # Getting the max historical Marker of Genome B
        for valueB in genomeB.edges:
            hMarkerMaxB = max(hMarkerMaxB, valueB.hMarker)

        # Iterate through Genome A and find hMarker Pairs to Genome B,
        # then randomly choose between either Genome A or Genome B
        # If the Genome A is NOT the Fittest Genome, append the Edges below the Max hMarker Value of Genome B
        # and throw the excess away.
        for valueA in genomeA.edges:
            if genomeB.edges.__contains__(valueA.hMarker):
                if random.randint(0, 1) == 0:
                    fittestGenome.edges[np.where(fittestGenome.edges, valueA)] = valueA
                else:
                    fittestGenome.edges[np.where(fittestGenome.edges, valueA)] = genomeB.edges[
                        np.where(genomeB.edges, valueA)]
            else:
                if genomeA.score < genomeB.score and valueA.hMarker <= hMarkerMaxB:
                    fittestGenome.edges.append(valueA)
            hMarkerMaxA = max(hMarkerMaxA, valueA.hMarker)

        # Iterate through Genome B  and if its NOT the fittest Genome, append every Edge with an lower hMarker value
        # then the hMarker Max Value from Genome A. The Excess get thrown away.
        if genomeB.score < genomeA:
            for valueB in genomeB.edges:
                if valueB <= hMarkerMaxA:
                    fittestGenome.edges.append(valueB)
