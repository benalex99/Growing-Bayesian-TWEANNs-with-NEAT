import random
import time

from NeuroEvo.Genome.NodeGene import NodeGene
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome


class NEAT:
    population: list

    def __init__(self, iterations, batchSize, maxPopSize, episodeDur, showProgress=(0, 0)):
        self.population = []
        self.species = []
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
        disjoint = False

        for index, value in enumerate(weakestGenome.edges):
            if value.hMarker == fittestGenome.edges[min(index, len(fittestGenome.edges) - 1)].hMarker and not disjoint:
                if random.randint(0, 1) < 1:
                    fittestGenome.edges[index] = value
            else:
                disjoint = True
                if firstGenome.fitness == secondGenome.fitness:
                    while value.toNr >= len(fittestGenome.nodes) or value.fromNr >= len(fittestGenome.nodes):
                        fittestGenome.nodes.append(NodeGene(nodeNr=len(fittestGenome.nodes)))
                    if (fittestGenome.nodes[value.fromNr].layer <= fittestGenome.nodes[value.toNr].layer and
                            (not fittestGenome.nodes[value.fromNr].outputtingTo.__contains__(value.toNr))):
                        fittestGenome.edges.append(value)
                        fittestGenome.nodes[value.fromNr].outputtingTo.append(value.toNr)
                        fittestGenome.increaseLayers(fittestGenome.nodes[value.fromNr], fittestGenome.nodes[value.toNr])

        return fittestGenome

    def speciation(self, population, excessImp, disjointImp, weightImp):
        compareGenome = population[0]
        self.species.append([compareGenome])

        # The compare Genome to specify if the Genome should be in the same Species or create a new Species
        for index, genome in enumerate(population):
            if not (genome == compareGenome):
                factorN = self.setFactorN(compareGenome.edges, genome.edges)
                deltaValues = []

                # Checks the delta value for the representative of the species
                for species in self.species:
                    compareGenome = species[0]
                    if not (genome == compareGenome):
                        # Defines the number of Excess and Disjoint Genes and the average Weight difference
                        excessGenes, disjointGenes, avgWeight = self.getExcessAndDisjoint(genome, compareGenome)
                        deltaValues.append(
                            self.deltaValue(excessImp, disjointImp, weightImp, excessGenes, disjointGenes, avgWeight,
                                            factorN))

                if min(deltaValues) < 2:
                    self.species[deltaValues.index(min(deltaValues))].append(genome)
                else:
                    self.species.append([genome])

    # Checks if both Genomes have more then 20 Genes and set the nFactor to the length
    # of the Genome with more genes, if both are under 20 the nFactor is 1
    @staticmethod
    def setFactorN(firstGenomeEdges, secondGenomeEdges):
        n = max(len(firstGenomeEdges), len(secondGenomeEdges))
        if n > 20:
            return n
        else:
            return 1

    @staticmethod
    def deltaValue(excessImp, disjointImp, weightImp, excessGenes, disjointGenes, avgWeight, factorN):
        return (((excessImp * excessGenes) / factorN)
                + ((disjointImp * disjointGenes) / factorN)
                + (weightImp * avgWeight))

    @staticmethod
    def getExcessAndDisjoint(firstGenome, secondGenome):
        disjointGenes, excessGenes, avgWeight = 0, 0, 0

        if len(firstGenome.edges) == 0 or len(secondGenome.edges) == 0:
            excessGenes = max(len(firstGenome.edges) - 1, len(secondGenome.edges) - 1)
            return excessGenes, disjointGenes, avgWeight

        for index, edge in enumerate(secondGenome.edges):
            if not firstGenome.edges[min(index, len(firstGenome.edges) - 1)].hMarker == edge.hMarker:
                if edge.hMarker <= firstGenome.edges[len(firstGenome.edges) - 1].hMarker:
                    disjointGenes += 1
                else:
                    excessGenes += 1
            else:
                # Defines the average Weight of matching Gene
                if not index > len(firstGenome.edges) - 1:
                    avgWeight += abs(edge.weight - firstGenome.edges[index].weight)
                    if not firstGenome.edges[min(index + 1, len(firstGenome.edges) - 1)].hMarker \
                           == edge.hMarker:
                        avgWeight = avgWeight / (index + 1)

        for indexEdges, edge in enumerate(firstGenome.edges):
            if not edge.hMarker == secondGenome.edges[min(indexEdges, len(secondGenome.edges) - 1)].hMarker:
                if edge.hMarker <= secondGenome.edges[len(secondGenome.edges) - 1].hMarker:
                    disjointGenes += 1
                else:
                    excessGenes += 1
        return excessGenes, disjointGenes, avgWeight

    def bestGene(self):
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        bestGene = self.population[0]
        for genome in self.population:
            if genome.fitness > bestGene.fitness:
                bestGene = genome
        return bestGene

    def visualize(self, gene, env, duration, useDone=True, seed=0):
        gene.visualize()
        env.visualize(gene, duration=duration, useDone=useDone, seed=seed)

    def newGenome(self):
        if (random.randint(0, 1) < 1 or len(self.population) <= 3):
            g = self.population[random.randint(0, max(0, int(len(self.population) / 2 - 1)))].copy()
            self.hMarker += g.mutate(self.hMarker)
        else:
            g1 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            g2 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            while (g1 == g2):
                g2 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            g = self.merge(self.population[g1],
                           self.population[g2])
            self.speciation(self.population, 0, 0, 0)
        return g
