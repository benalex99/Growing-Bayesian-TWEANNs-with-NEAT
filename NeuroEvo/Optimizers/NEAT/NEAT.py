import copy
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

    def run(self, rootGenome, env):
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
            while len(self.population) > self.maxPopSize:
                i = random.randint(max(0, int(len(self.population) / 2 - 1)), len(self.population) - 1)
                self.population.remove(self.population[i])
            self.median = self.population[int(len(self.population) / 2)].fitness

            print("Took : " + str(time.time() - ntime))
            print("Median score: " + str(self.median))
            print("Best Score: " + str(self.bestGene().fitness) + "\n")
            if self.showProgress[0] > 0 and iter % self.showProgress[0] == 0:
                self.visualize(self.bestGene(), env, self.showProgress[1], seed=iter)

        # Return the best gene from the population
        return self.bestGene()

    @staticmethod
    def merge(firstGenome, secondGenome):
        """The merging process takes both parent Genomes and merge it together. Genes with the same historical Marker
        get randomly chosen by one of the parent Genome. Any other Disjoint or Excess Genes get added to the merged
        Genome

        Args:
            firstGenome (NEATGenome): Parent Genome 1
            secondGenome (NEATGenome): Parent Genome 2

        Returns:
            NEATGenome: The merged child Genome of both parent Genomes
        """
        fitterGenome = firstGenome.copy() if firstGenome.fitness >= secondGenome.fitness else secondGenome.copy()
        weakerGenome = firstGenome.copy() if firstGenome.fitness < secondGenome.fitness else secondGenome.copy()
        disjoint = False

        # Iterates through the Genome with the lower Fitness Score to implement missing Genes to the fitter Genome
        for index, edge in enumerate(weakerGenome.edges):

            if edge.hMarker == fitterGenome.edges[min(index, len(fitterGenome.edges) - 1)].hMarker and not disjoint:
                print(edge.hMarker)
                if random.randint(0, 1) < 1:
                    # Randomly assign one of either genomes weights
                    fitterGenome.edges[index].weight = edge.weight
            else:
                disjoint = True
                if firstGenome.fitness == secondGenome.fitness:
                    # if the receiving or sending node does not exist, add it
                    # TODO: Adding redundant nodes?
                    while edge.toNr >= len(fitterGenome.nodes) or edge.fromNr >= len(fitterGenome.nodes):
                        fitterGenome.nodes.append(NodeGene(nodeNr=len(fitterGenome.nodes)))
                    # Check if the receiving neuron is not in a lower or equal layer
                    # And if the connection already exists
                    # TODO: Nodes has standard 0 layer assignment
                    if (fitterGenome.nodes[edge.fromNr].layer <= fitterGenome.nodes[edge.toNr].layer and
                            (not fitterGenome.nodes[edge.fromNr].outputtingTo.__contains__(edge.toNr))):
                        fitterGenome.edges.append(copy.deepcopy(edge))
                        if edge.enabled:
                            fitterGenome.nodes[edge.fromNr].outputtingTo.append(edge.toNr)
                        fitterGenome.increaseLayers(fitterGenome.nodes[edge.fromNr], fitterGenome.nodes[edge.toNr])

        return fitterGenome

    def speciation(self, population, excessImp, disjointImp, weightImp, inclusionThreshold=2):
        """Compares all genomes from a population with the representative genome of a species and append or defines new
        Species.

        Args:
            population (list): All Genomes
            excessImp (float): The Importance of all Excess Genes within two Genomes
            disjointImp (float): The Importance of all Disjoint Genes within two Genomes
            weightImp (float): The Importance of the Weight difference between two Genomes
            inclusionThreshold (float): The Range where the Delta of an Genome to the compare Genome should be appended
                                        to the Species of the compare Genome

        Returns:
            None
        """
        compareGenome = population[0]
        self.species = []
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
                        disjointGenes, excessGenes, avgWeight = self.getExcessAndDisjoint(genome, compareGenome)
                        deltaValues.append(
                            self.deltaValue(excessImp, disjointImp, weightImp, excessGenes, disjointGenes, avgWeight,
                                            factorN))

                if min(deltaValues) < inclusionThreshold:
                    self.species[deltaValues.index(min(deltaValues))].append(genome)
                else:
                    self.species.append([genome])

    @staticmethod
    def setFactorN(firstGenomeEdges, secondGenomeEdges):
        """Checks if one Genomes have more then 20 Genes and set the 'Factor N' (Normalize Function)
        to the length of the Genome with more Genes. If both Genomes have under 20 Genes, the 'Factor N' is set to 1.

        Args:
            firstGenomeEdges (list): All Edges from the Genomes we compare to
            secondGenomeEdges (list): All Edges from the current Genome

        Returns:
            (int): The size of all Genes in the larger Genome if one Genome have more then 20 Genes, if not return 1
        """
        n = max(len(firstGenomeEdges), len(secondGenomeEdges))
        if n > 20:
            return n
        else:
            return 1

    @staticmethod
    def deltaValue(excessImp, disjointImp, weightImp, excessGenes, disjointGenes, avgWeight, factorN):
        """Calculates the Delta Value of two Genomes compared to each other based on Excess Genes, Disjoint Genes
        and the Weight difference

        Args:
            excessImp (float): The Importance of all Excess Genes within two Genomes
            disjointImp (float): The Importance of all Disjoint Genes within two Genomes
            weightImp (float): The Importance of the Weight difference between two Genomes
            excessGenes (int): The Number of all Excess Genes within two Genomes
            disjointGenes (int): The Number of all Disjoint Genes within two Genomes
            avgWeight (float): The average Weight between matching Genes
            factorN (int): Normalizes function for Genome Size
        Returns:
            The calculated Delta Value as an float.
        """
        return (((excessImp * excessGenes) / factorN)
                + ((disjointImp * disjointGenes) / factorN)
                + (weightImp * avgWeight))

    @staticmethod
    def getExcessAndDisjoint(firstGenome, secondGenome):
        """Gets the number of all Excess and Disjoint Genes within both Genomes. The Average Weight of matching Genes
        is calculated

        Args:
            firstGenome (NEATGenome): The current Genome
            secondGenome (NEATGenome): The compare Genome

        Returns:
            (int, int, float): The Number of all Disjoint Genes, all Excess Genes and the average Weight of matching Genes
        """
        disjointGenes, excessGenes, avgWeight = 0, 0, 0.0

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
        return disjointGenes, excessGenes, avgWeight

    def bestGene(self):
        """Iterate through the whole Population and look for the fittest Genome

        Returns:
            NEATGenome: Returns the Genome with the best Fitness Value
        """
        # Sorts the Population by the Genome with the highest Fitness Value
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return self.population[0]

    @staticmethod
    def visualize(gene, env, duration, useDone=True, seed=0):
        gene.visualize()
        env.visualize(gene, duration=duration, useDone=useDone, seed=seed)

    def newGenome(self):
        if random.randint(0, 1) < 1 or len(self.population) <= 3:
            g = self.population[random.randint(0, max(0, int(len(self.population) / 2 - 1)))].copy()
            self.hMarker += g.mutate(self.hMarker)
        else:
            g1 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            g2 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            while (g1 == g2):
                g2 = random.randint(0, max(0, int(len(self.population) / 2 - 1)))
            g = self.merge(self.population[g1],
                           self.population[g2])
            # self.speciation(self.population, 0, 0, 0)
        return g
