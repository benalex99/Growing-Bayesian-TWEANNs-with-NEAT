import random
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome
import numpy as np
import time

class NEAT:
    population: list

    def __init__(self, iterations, batchSize, maxPopSize):
        self.population = []
        self.iterations = iterations
        self.batchSize = batchSize
        self.maxPopSize = maxPopSize
        return

    def run(self, rootGenome: NEATGenome, env):
        self.hMarker = 1
        self.population.append(rootGenome)

        for iter in range(self.iterations):
            print("Iteration: " + str(iter))
            ntime = time.time()
            avgSize = 0
            toBeTested = []
            # Create a bunch of mutations
            for i in range(self.batchSize):
                g: NEATGenome = self.population[random.randint(0, len(self.population) - 1)].copy()
                self.hMarker = self.hMarker + g.mutate(self.hMarker)
                avgSize += len(g.edges)
                toBeTested.append(g)
            print("AvgSize : " + str(avgSize/len(toBeTested)))
            # Assign values to the mutations and add them to the population
            env.test(toBeTested)
            self.population = self.population + toBeTested
            self.population.sort(key= lambda x: x.fitness)
            self.median = self.population[int(len(self.population)/2)].fitness
            # Discard bad mutations from the population until the max population count is reached
            while (len(self.population) > self.maxPopSize):
                i = random.randint(0, len(self.population) - 1)
                if (self.population[i].fitness <= self.median):
                    self.median = self.population[int(len(self.population)/2)].fitness
                    self.population.remove(self.population[i])
            print("Took : " + str(time.time() - ntime))
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
        if genomeA.fitness > genomeB.fitness:
            fittestGenome = genomeA
        else:
            fittestGenome = genomeB
        # fittestGenome = lambda x: genomeA if genomeA.fitness > genomeB.fitness else genomeB   -> Just to looks nice
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
                if genomeA.fitness < genomeB.fitness and valueA.hMarker <= hMarkerMaxB:
                    fittestGenome.edges.append(valueA)
                    # Checks if the two Nodes from the appended Edge of Genome A exists in the Fittest Genome
                    # If NOT, then append it to Nodes
                    if not fittestGenome.nodes.__contains__(valueA.fromNr):
                        fittestGenome.nodes.append(genomeA.nodes[np.where(genomeA.nodes, valueA.fromNr)])
                    elif not fittestGenome.nodes.__contains__(valueA.toNr):
                        fittestGenome.nodes.append(genomeA.nodes[np.where(genomeA.nodes, valueA.toNr)])
            hMarkerMaxA = max(hMarkerMaxA, valueA.hMarker)

        # Iterate through Genome B  and if its NOT the fittest Genome, append every Edge with an lower hMarker value
        # then the hMarker Max Value from Genome A. The Excess get thrown away.
        if genomeB.fitness < genomeA:
            for valueB in genomeB.edges:
                if valueB <= hMarkerMaxA:
                    fittestGenome.edges.append(valueB)
                    # Checks if the two Nodes from the appended Edge of Genome B exists in the Fittest Genome
                    # If NOT, then append it to Nodes
                    if not fittestGenome.nodes.__contains__(valueB.fromNr):
                        fittestGenome.nodes.append(genomeB.nodes[np.where(genomeB.nodes, valueB.fromNr)])
                    elif not fittestGenome.nodes.__contains__(valueB.toNr):
                        fittestGenome.nodes.append(genomeB.nodes[np.where(genomeB.nodes, valueB.toNr)])
