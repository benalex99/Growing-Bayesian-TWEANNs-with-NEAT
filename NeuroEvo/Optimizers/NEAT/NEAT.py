import random

class NEAT:

    def __init__(self, env):
        self.population = []
        self.env = env
        return

    def run(self, rootGenome, iterations):

        self.population.append(rootGenome)
        for iter in range(iterations):
            self.population[random.randint(0,len(self.population)-1)].mutate()