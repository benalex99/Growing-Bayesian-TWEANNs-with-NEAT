import copy
import math
import random
import time

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome


class ProbabilisticNEAT:
    population: list

    def __init__(self, iterations, maxPopSize, batchSize, episodeDur=400, showProgress=(0, 0),
                 excessImp=0.5, disjointImp=0.5, weightImp=1, inclusionThreshold=5,
                 useMerging=True, useSpeciation=True, weightsOnly=False):
        """

        iterations (int):
            maxPopSize (int):
            batchSize (int):
            episodeDur (int):
            showProgress int, int:
            excessImp (float): The Importance of all Excess Genes within two Genomes
            disjointImp (float): The Importance of all Disjoint Genes within two Genomes
            weightImp (float): The Importance of the Weight difference between two Genomes
            inclusionThreshold (float): The Range where the Delta of an Genome to the compare Genome should be appended
                                        to the Species of the compare Genome
        """
        self.population = []
        self.maxPopSize = maxPopSize
        self.species = []
        self.iterations = iterations
        self.batchSize = batchSize
        self.episodeDur = episodeDur
        self.showProgress = showProgress
        self.excessImp = excessImp
        self.disjointImp = disjointImp
        self.weightImp = weightImp
        self.inclusionThreshold = inclusionThreshold
        self.hMarker = 1
        self.useMerging = useMerging
        self.useSpeciation = useSpeciation
        self.weightsOnly = weightsOnly
        self.maxFitnesses = []
        return

    def run(self, rootGenome, env):
        self.population.append(rootGenome)
        self.visualize(rootGenome, env, 500, useDone=False)

        for iteration in range(self.iterations):
            print("Iteration: " + str(iteration))
            ntime = time.time()

            # Assign values to the mutations and add them to the population
            env.test(self.population, self.episodeDur, seed=iteration)

            if self.useSpeciation:
                self.speciation()
                self.sharingFitness()
            self.murderGenomes(self.useSpeciation)
            self.mutationBasedOnSharedFitness(self.useSpeciation)

            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.median = self.population[int(len(self.population) / 2) - 1].fitness

            print("Took : " + str(time.time() - ntime))
            print("Median score: " + str(self.median))
            print("Best Score: " + str(self.bestGene().fitness))
            counts = []
            for speciesEntry in self.species:
                counts.append(len(speciesEntry))
            print("Batch Size: " + str(len(self.population)))
            print("Population Size: " + str(len(self.population)+sum(counts)))
            print("Species: " + str(len(self.species)))
            print("Species Sum: " + str(sum(counts)))
            print("Species counts: " + str(counts) + "\n")
            if self.showProgress[0] > 0 and iteration % self.showProgress[0] == 0:
                self.visualize(self.bestGene(), env, self.showProgress[1], seed=iteration)

        # Return the best gene from the population
        return self.bestGene()

    def debug(self):
        counts = []
        for speciesEntry in self.species:
            counts.append(len(speciesEntry))
        print("Population Size: " + str(len(self.population)))
        print("Species: " + str(len(self.species)))
        print("Species Sum: " + str(sum(counts)))
        print("Species counts: " + str(counts) + "\n")

    def murderGenomes(self, useSpeciation):
        """

        :return:
        """

        if useSpeciation:
            counts = []
            for speciesEntry in self.species:
                counts.append(len(speciesEntry))
            count = sum(counts)

            self.population.sort(key=lambda x: x.adjustedFitness, reverse=True)
            while count > self.maxPopSize:
                genome = self.population.pop(len(self.population) - 1)
                count -= 1
                for speciesEntry in self.species:
                    if speciesEntry.__contains__(genome):
                        if len(speciesEntry) > 1:
                            for genomeEntry in speciesEntry:
                                genomeEntry.adjustedFitness *= len(speciesEntry) / (len(speciesEntry) - 1)
                        speciesEntry.remove(genome)
                self.population.sort(key=lambda x: x.adjustedFitness, reverse=True)

            # Remove empty species from the species list
            murderedSpecies = []
            for species in self.species:
                if len(species) == 0:
                    murderedSpecies.append(species)

            for species in murderedSpecies:
                self.maxFitnesses.pop(self.species.index(species))
                self.species.remove(species)
            for index, (species, maxFitness) in enumerate(zip(self.species, self.maxFitnesses)):
                if maxFitness[1] > 15:
                    self.maxFitnesses.pop(index)
                    self.species.pop(index)
        else:
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            while len(self.population) > self.maxPopSize:
                self.population.pop(len(self.population) - 1)

    def mutationBasedOnSharedFitness(self, useSpeciation):
        """Mutates random Genomes in the Species in range of the percentage of the summed Fitness of all Genomes in
        a species.

        Args:

        Returns:
            None
        """

        mutations = []
        if useSpeciation:
            mutationIterations = self.calculateMutations()
            for species, iterations in zip(self.species, mutationIterations):
                for _ in range(iterations):
                    mutations.append(self.newGenome(species[random.randint(0, len(species) - 1)]))
            self.population = mutations
        else:
            for _ in range(self.batchSize):
                mutations.append(self.newGenome(self.population[random.randint(0, len(self.population) - 1)]))
            self.population.extend(mutations)

    def calculateMutations(self):
        """Calculates how many Mutation each Species gets

        Args:

        Returns:
            list: A List of the amount of Mutations each Species get
        """
        summedFitnesses = self.sumFitnessValue()
        mutationNumbers = []
        if(sum(summedFitnesses) > 0):
            for index, fitness in enumerate(summedFitnesses):
                mutationNumbers.append(fitness)
                mutationNumbers[index] *= self.batchSize / sum(summedFitnesses)
                mutationNumbers[index] = math.floor(mutationNumbers[index])
        else:
            for index, fitness in enumerate(summedFitnesses):
                mutationNumbers.append(fitness)
                mutationNumbers[index] *= self.batchSize / len(summedFitnesses)
                mutationNumbers[index] = math.floor(mutationNumbers[index])

        # Randomly allocate mutations that were lost due to rounding
        while self.batchSize > sum(mutationNumbers):
            index = random.randint(0, len(self.species) - 1)
            mutationNumbers[index] += 1

        return mutationNumbers

    def sumFitnessValue(self):
        """Sums up each Fitness Values in every Species

        Args:

        Returns:
            list: The summed Fitness Values for all Genome in each Species separated
        """
        summedFitness = []
        for index, entrySpecies in enumerate(self.species):
            summedFitness.append(0)
            for genome in entrySpecies:
                summedFitness[index] += genome.adjustedFitness
        return summedFitness

    @staticmethod
    def merge(firstGenome, secondGenome):
        """The merging process takes both parent Genomes and merge it together. Genes with the same historical Marker
        get randomly chosen by one of the parent Genome. Any other Disjoint or Excess Genes get added to the merged
        Genome

        Args:
            firstGenome (ProbabilisticGenome): Parent Genome 1
            secondGenome (ProbabilisticGenome): Parent Genome 2

        Returns:
            ProbabilisticGenome: The merged child Genome of both parent Genomes
        """
        fitterGenome = firstGenome.copy() if firstGenome.fitness >= secondGenome.fitness else secondGenome.copy()
        weakerGenome = firstGenome.copy() if firstGenome.fitness < secondGenome.fitness else secondGenome.copy()
        disjoint = False

        # Iterates through the Genome with the lower Fitness Score to carry over weights
        for index, edge in enumerate(weakerGenome.edges):
            if len(fitterGenome.edges) == 0:
                break
            if edge.hMarker == fitterGenome.edges[min(index, len(fitterGenome.edges) - 1)].hMarker and not disjoint:
                if random.randint(0, 1) < 1:
                    # Randomly assign one of either genomes weights
                    fitterGenome.edges[index].weight = edge.weight
            else:
                disjoint = True
                # For disjoint and excess edges,
                if firstGenome.fitness == secondGenome.fitness:
                    # if the receiving or sending node does not exist, add it
                    # TODO: Adding redundant nodes?
                    # while edge.toNr >= len(fitterGenome.nodes) or edge.fromNr >= len(fitterGenome.nodes):
                    #     fitterGenome.nodes.append(NodeGene(nodeNr=len(fitterGenome.nodes)))

                    # Lack of clarity in paper. Random carrying over results in cycles.
                    # We only carry over edges for which nodes already exist, to ensure alignment and connectivity
                    if edge.toNr >= len(fitterGenome.nodes) or edge.fromNr >= len(fitterGenome.nodes):
                        continue
                    # Check if classes align
                    if edge.fromClass != None:
                        if fitterGenome.nodes[edge.fromNr].classCount < edge.fromClass:
                            continue
                    # Check if classes align
                    if edge.toClass != None:
                        if fitterGenome.nodes[edge.toNr].classCount < edge.toClass:
                            continue

                    # Check if the receiving neuron is not in a lower or equal layer
                    # And if the connection already exists
                    # TODO: New nodes have standard 0 layer assignment
                    if (fitterGenome.nodes[edge.fromNr].layer <= fitterGenome.nodes[edge.toNr].layer or
                            not fitterGenome.edgeExists(edge)):
                        continue

                    fitterGenome.edges.append(copy.deepcopy(edge))
                    if edge.enabled:
                        fitterGenome.nodes[edge.fromNr].outputtingTo.append(edge.toNr)
                    fitterGenome.increaseLayers(fitterGenome.nodes[edge.fromNr], fitterGenome.nodes[edge.toNr])

        return fitterGenome

    def speciation(self):
        """Compares all genomes from a population with the representative genome of a species and append or defines new
        Species.

        Args:

        Returns:
            None
        """

        newSpecies = []
        # Reassign Species representatives
        for species in self.species:
            genome = species[random.randint(0, len(species) - 1)]
            newSpecie = [genome]
            if len(species) > 5:
                species.sort(key=lambda x: x.fitness, reverse=True)
                newSpecie.append(species[0])
            newSpecies.append(newSpecie)
        self.species = newSpecies

        # The compare Genome to specify if the Genome should be in the same Species or create a new Species
        for index, genome in enumerate(self.population):
            speciesFound = False

            # Checks the delta value for the representative of the species
            for speciesIndex, species in enumerate(self.species):
                compareGenome = species[0]
                if not (genome == compareGenome):
                    # Defines the number of Excess and Disjoint Genes and the average Weight difference
                    deltaValue = self.deltaValue(genome, compareGenome)

                    # Assign Genome to first matching Species
                    if deltaValue < self.inclusionThreshold:
                        speciesFound = True
                        species.append(genome)
                        break
                else:
                    speciesFound = True
                    break

            if not speciesFound:
                self.species.append([genome])
                self.maxFitnesses.append([genome.fitness, 0])

        # Update stagnation
        for species, maxFitness in zip(self.species, self.maxFitnesses):
            species.sort(key=lambda x: x.fitness, reverse=True)
            if maxFitness[0] < species[0].fitness:
                maxFitness[0] = species[0].fitness
                maxFitness[1] = 0
            else:
                maxFitness[1] += 1

    @staticmethod
    def setFactorN(firstGenomeEdges, secondGenomeEdges):
        """Checks if one Genome have more then 20 Genes and set the 'Factor N' (Normalize Function)
        to the length of the Genome with more Genes. If both Genomes have under 20 Genes, the 'Factor N' is set to 1.

        Args:
            firstGenomeEdges (list): All Edges from the Genomes we compare to
            secondGenomeEdges (list): All Edges from the current Genome

        Returns:
            (int): Returns N if one Genome have more then 20 Genes, or 1 if both have less
        """
        n = max(len(firstGenomeEdges), len(secondGenomeEdges))
        if n > 20:
            return n
        else:
            return 1

    def deltaValue(self, firstGenome, secondGenome):
        """Calculates the Delta Value of two Genomes compared to each other based on Excess Genes, Disjoint Genes
        and the Weight difference

        Args:
            firstGenome (ProbabilisticGenome): f
            secondGenome (ProbabilisticGenome): d
            excessImp (float): The Importance of all Excess Genes within two Genomes
            disjointImp (float): The Importance of all Disjoint Genes within two Genomes
            weightImp (float): The Importance of the Weight difference between two Genomes

        Returns:
            The calculated Delta Value as an float.
        """
        excessGenes, disjointGenes, avgWeight = self.getExcessAndDisjoint(firstGenome, secondGenome)
        factorN = self.setFactorN(firstGenome.edges, secondGenome.edges)

        return (((self.excessImp * excessGenes) / factorN)
                + ((self.disjointImp * disjointGenes) / factorN)
                + (self.weightImp * avgWeight))

    @staticmethod
    def getExcessAndDisjoint(firstGenome, secondGenome):
        """Gets the number of all Excess and Disjoint Genes within both Genomes. The Average Weight of matching Genes
        is calculated

        Args:
            firstGenome (ProbabilisticGenome): The current Genome
            secondGenome (ProbabilisticGenome): The compare Genome

        Returns:
            (int, int, float): The Number of all Disjoint Genes, all Excess Genes and the average Weight of matching Genes
        """
        disjointGenes, excessGenes, avgWeight = 0, 0, 0.0

        if len(firstGenome.edges) == 0 or len(secondGenome.edges) == 0:
            excessGenes = max(len(firstGenome.edges) - 1, len(secondGenome.edges) - 1)
            return excessGenes, disjointGenes, avgWeight

        for index, edge in enumerate(secondGenome.edges):
            try:
                if not firstGenome.edges[index].hMarker == edge.hMarker:
                    if edge.hMarker <= firstGenome.edges[len(firstGenome.edges) - 1].hMarker:
                        disjointGenes += 1
                    else:
                        excessGenes += 1
                else:
                    # Defines the average Weight of matching Gene
                    if not index > len(firstGenome.edges) - 1:
                        avgWeight += abs(edge.weight - firstGenome.edges[index].weight)
                        if not firstGenome.edges[index + 1].hMarker == edge.hMarker:
                            avgWeight = avgWeight / (index + 1)
            except IndexError:
                pass

        for indexEdges, edge in enumerate(firstGenome.edges):
            if not edge.hMarker == secondGenome.edges[min(indexEdges, len(secondGenome.edges) - 1)].hMarker:
                if edge.hMarker <= secondGenome.edges[len(secondGenome.edges) - 1].hMarker:
                    disjointGenes += 1
                else:
                    excessGenes += 1
        return disjointGenes, excessGenes, avgWeight

    def sharingFitness(self):
        """Calculates the new fitness of each Genome based on the length of all Entry's in the Species

         Args:

         Returns:
             None
        """
        fitnesses = []
        for specie in self.species:
            for genome in specie:
                fitnesses.append(genome.fitness)

        for entrySpecies in self.species:
            for genome in entrySpecies:
                # Making all fitnesses positive with an offset
                genome.adjustedFitness = (genome.fitness - min(min(fitnesses) - 1, 0)) / len(entrySpecies)

    def bestGene(self):
        """Iterate through the whole Population and look for the fittest Genome

        Returns:
            ProbabilisticGenome: Returns the Genome with the best Fitness Value
        """
        allGenomes = []
        for genome in self.population:
            allGenomes.append(genome)
        for specie in self.species:
            for genome in specie:
                allGenomes.append(genome)

        # Sorts the Population by the Genome with the highest Fitness Value
        allGenomes.sort(key=lambda x: x.fitness, reverse=True)
        return allGenomes[0]

    @staticmethod
    def visualize(gene, env, duration, useDone=True, seed=0):
        gene.visualize()
        env.visualize(gene, duration=duration, useDone=useDone, seed=seed)

    def newGenome(self, parentGenome):
        if random.randint(0, 1) < 1 or len(self.population) <= 3 or not self.useMerging:
            g = parentGenome.copy()
            success = 0
            while success <= 0:
                success = g.mutate(self.hMarker, weightsOnly=self.weightsOnly)
            self.hMarker += success
        else:
            g1 = parentGenome
            g2 = self.population[random.randint(0, len(self.population) - 1)]
            while g1 == g2:
                g2 = self.population[random.randint(0, len(self.population) - 1)]
            g = self.merge(g1, g2)
        g.fitness = -math.inf
        return g
