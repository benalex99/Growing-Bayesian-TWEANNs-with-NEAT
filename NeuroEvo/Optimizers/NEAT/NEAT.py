import random

class NEAT:

    def __init__(self, env):
        self.population = []
        self.env = env
        self.hMarker = 1
        return

    def run(self, rootGenome, iterations, batchSize, maxPopSize):

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

            # Discard bad mutations from the population until the max population count is reached
            while(len(self.population) > maxPopSize):
                i = random.randint(0,len(self.population) - 1)
                if(self.population)