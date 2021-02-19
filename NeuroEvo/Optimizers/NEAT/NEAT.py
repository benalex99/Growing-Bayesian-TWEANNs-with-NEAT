import random

class NEAT:

    population: list

    def __init__(self, env):
        self.population = []
        self.env = env
        self.hMarker = 1
        return

    def testRun(self, rootGenome, iterations, batchSize, maxPopSize):

        self.population.append(rootGenome)
        toBeTested = []
        for iter in range(iterations):
            # Create a bunch of mutations
            for i in range(batchSize):
                g = self.population[random.randint(0,len(self.population)-1)].copy()
                self.hMarker = self.hMarker + g.mutate(self.hMarker)
                toBeTested.append(g)

            # Assign values to the mutations and add themto the population
            self.env.test(toBeTested)
            self.population =+ toBeTested
            self.avgScore = 0
            for genome in self.population:
                self.avgScore += genome.fitness
            self.avgScore = self.avgScore / len(self.population)

            # Discard bad mutations from the population until the max population count is reached
            while(len(self.population) > maxPopSize):
                i = random.randint(0,len(self.population) - 1)
                if(self.population[i] < self.avgScore):
                    self.population.remove(i)

        # Get the best gene from the population
        bestGene = self.population[0]
        for genome in self.population:
            if(genome.fitness > bestGene.fitness):
                bestGene = genome

        return bestGene