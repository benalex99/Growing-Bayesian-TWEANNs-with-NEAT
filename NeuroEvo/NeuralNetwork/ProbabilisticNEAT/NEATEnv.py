import math

from matplotlib import pyplot as plt
from tqdm import tqdm

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
import torch



class VariableEnv:
    def __init__(self, model = ProbabilisticGenome(1, 1), datapointCount = 1000, detail=20):
        self.model = model

        # Generate a distribution
        self.input = torch.ones((datapointCount, model.inputSize), device=torch.device('cuda'))
        self.generated = self.model.generate(self.input)
        self.data = [self.input, self.generated]

        # Set the clustering detail for the discrete KL-divergence
        self.detail = detail

    def test(self, population, duration=None, seed=None):
        tests = []
        for genome in population:
            if genome.fitness == -math.inf:
                tests.append(genome)
        for genome in tqdm(tests):
            predictions = genome.generate(self.input)
            genome.fitness = -VariableEnv.mse_loss(self.generated, predictions, detail=self.detail)

    def testOne(self, genome):
        predictions = genome.generate(self.input)
        return -VariableEnv.mse_loss(self.generated, predictions, detail=self.detail)

    @staticmethod
    def discretizedKullbackLeibler(targets, predictions, detail=20, punishIgnorance=False):
        """
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail at which we compare. Determines the amount of buckets. Can be any positive real.
        :param punishIgnorance: Whether to punish a disjoint probability space
                                    (Samples from P that have a probability of 0 according to Q)
        :return: The approximated Kullback-Leibler divergence KL(P||Q)
        """

        p, q = VariableEnv.discretizedDistributionProbs(targets, predictions, detail)

        # Calculate the Kullback-Leibler divergence KL(P||Q)
        klDivergence = 0
        for x in q.keys():
            if p.keys().__contains__(x):
                klDivergence += p[x] * math.log(p[x]/q[x])
        #
        # # TODO: What about values for P that are not supported by Q?
        # #  If we do not punish this, the model will attempt to
        # #  optimize itself for having a disjoint probability space
        # # Suggestion: multiply final KL-divergence by 1+lostPProb. Worst case, KL divergence is doubled.
        # if(punishIgnorance):
        #     lostPProb = 0
        #     for x in p.keys():
        #         if not q.keys().__contains__(x):
        #             lostPProb += p[x]
        #     klDivergence *= 1+lostPProb
        #     # print("lostProb " + str(lostPProb))
        #     #                                                           __           __
        #     # If all predictions are disjoint, there is nothing we can do \ ( ^-^ ) /
        #     if lostPProb >= 1:
        #         klDivergence = 1000000 # A large value, because neat doesnt handle infinity well

        return klDivergence

    @staticmethod
    def mse_loss(predictions, targets, detail=20):
        '''
        We do this because KL-divergence is not suited for structure optimization. KL-divergence does not handle distributions
        with differing supports.
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail at which we compare. Determines the amount of buckets. Can be any positive real.
        :return: The mse_loss of the prediction and target class probability distributions
        '''
        p, q = VariableEnv.discretizedDistributionProbs(targets, predictions, detail)
        # Calculate the MSE-loss
        loss = 0
        count = 0
        for x in list(q.keys()) + list(p.keys()):
            count += 1
            if p.keys().__contains__(x) and q.keys().__contains__(x):
                loss += (q[x] - p[x]) ** 2
            elif q.keys().__contains__(x):
                loss += (q[x] - 0) ** 2
            else:
                loss += (0 - p[x]) ** 2
        return loss / count

    @staticmethod
    def discretizedDistributionProbs(targets, predictions, detail):
        '''
        It batches continuous variables together to form classes.

        It normalizes the continuous data to a range of 1, scales the values up by the detail parameter
        and rounds to the closest integer. These integers then represent class indices.
        Then the class occurrences are counted and their probabilities calculated.
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail at which we compare. Determines the amount of classes. Can be any positive real.
        :return: Discretize probability distributions p and q
        '''

        # Normalize each dimension across all samples.
        max = torch.stack([predictions, targets]).max(0)[0][0]
        min = torch.stack([predictions, targets]).min(0)[0][0]
        diff = max - min
        diff[diff==0] = 1 # Don't divide by zero
        predNormed = predictions / diff
        targetsNormed = targets / diff

        # Scale up and round to closest integers, so we have up to D * detail different buckets
        predNormed = torch.round(predNormed * detail)
        targetsNormed = torch.round(targetsNormed * detail)

        # Count the occurrences in each bucket
        uniquePreds = {}
        for pred in predNormed:
            key = str(pred.tolist())
            if uniquePreds.keys().__contains__(key):
                uniquePreds[key] += 1
            else:
                uniquePreds[key] = 1
        uniqueTargets = {}
        for target in targetsNormed:
            key = str(target.tolist())
            if uniqueTargets.keys().__contains__(key):
                uniqueTargets[key] += 1
            else:
                uniqueTargets[key] = 1

        # Calculate the probabilities of the values based on the occurrences
        q = {}
        p = {}
        for pred in uniquePreds.keys():
            q[pred] = uniquePreds[pred] / len(predictions)

        for target in uniqueTargets.keys():
            p[target] = uniqueTargets[target] / len(targets)

        return p, q

    def inputs(self):
        return self.model.inputSize

    def outputs(self):
        return self.model.outputSize

    def visualize(self, gene, duration, useDone=None, seed=None):
        pass
        # gene.visualize()
        # plt.pause(duration/100)
        # self.model.visualize()
        # plt.pause(duration/100)