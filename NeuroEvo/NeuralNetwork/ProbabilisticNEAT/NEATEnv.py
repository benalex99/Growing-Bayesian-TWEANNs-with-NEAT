import math

from matplotlib import pyplot as plt
from tqdm import tqdm

from NeuroEvo.NeuralNetwork.ProbabilisticNEAT.ProbabilisticGenome import ProbabilisticGenome
import torch



class VariableEnv:
    def __init__(self, model = ProbabilisticGenome(1, 1), datapointCount = 1000, detail=20, criterion="me"):
        self.model = model
        self.criterion = criterion

        # Generate a distribution
        self.input = torch.ones((datapointCount, model.inputSize), device=torch.device('cuda'))
        self.generated = self.model.generate(self.input)
        self.data = [self.input, self.generated]

        # Set the clustering detail for the discrete KL-divergence
        self.detail = detail

    def test(self, population, duration=None, seed=None):
        tests = []
        # We only want to calculate the score for those models that have not yet received a score
        # So grab out the ones that have -inf (default) score
        for genome in population:
            if genome.fitness == -math.inf:
                tests.append(genome)
        # Test those that have just been filtered out
        for genome in tests:
            predictions = genome.generate(self.input)
            if self.criterion == "KL":
                genome.fitness = -VariableEnv.discretizedKullbackLeibler(self.generated, predictions, detail=self.detail)
            elif self.criterion == "me":
                genome.fitness = -VariableEnv.me_loss(self.generated, predictions, detail=self.detail)
            elif self.criterion == "mePenalized":
                genome.fitness = -VariableEnv.penalizedMe_loss(genome, self.generated, predictions, detail=self.detail)

    def testOne(self, genome):
        predictions = genome.generate(self.input)
        return -VariableEnv.me_loss(predictions, self.generated, detail=self.detail)

    @staticmethod
    def discretizedKullbackLeibler(targets, predictions, detail=20, punishIgnorance=False):
        """
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail with which we discretize the distributions. Determines the amount of buckets.
                        Can be any positive real.
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

        # Suggestion: multiply final KL-divergence by 1+lostPProb. Worst case, KL divergence is doubled.
        if(punishIgnorance):
            lostPProb = 0
            for x in p.keys():
                if not q.keys().__contains__(x):
                    lostPProb += p[x]
            klDivergence *= 1+lostPProb
            #                                                                  __           __
            # If all predictions are non-overlapping, there is nothing we can do \ ( ^-^ ) /
            if lostPProb >= 1:
                klDivergence = 1000000  # A large value because NEAT doesnt handle infinity well

        return klDivergence

    @staticmethod
    def me_loss(predictions, targets, detail=20):
        '''
        We do this because KL-divergence is not suited for structure optimization. KL-divergence does not handle
        distributions with differing supports.
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail at which we discretize the distributions. Determines the amount of buckets. Can be any positive real.
        :return: The me_loss of the prediction and target class probability distributions
        '''
        p, q = VariableEnv.discretizedDistributionProbs(targets, predictions, detail)
        # Calculate the ME-loss
        loss = 0
        count = 0
        # for x in list(q.keys()) + list(p.keys()):
        for x in list(dict.fromkeys(list(q.keys()) + list(p.keys()))):
            count += 1
            if p.keys().__contains__(x) and q.keys().__contains__(x):
                loss += abs(q[x] - p[x])
            elif q.keys().__contains__(x):
                loss += abs(q[x] - 0)
            else:
                loss += abs(0 - p[x])
        return loss / 2 #count

    @staticmethod
    def penalizedMe_loss(model, predictions, targets, detail=20):
        '''
        This criterion also takes the amount of parameters into account. It penalizes larger models.
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail with which we discretize the distributions. Determines the amount of buckets.
                        Can be any positive real.
        :return: The mse_loss of the prediction and target class probability distributions
        '''
        mse = VariableEnv.me_loss(predictions, targets, detail)
        params = 0
        for edge in model.edges:
            if edge.enabled:
                params += 1
        return mse + params

    @staticmethod
    def discretizedDistributionProbs(targets, predictions, detail):
        '''
        It batches continuous variables together to form classes.

        It normalizes the continuous data to a range of 1, scales the values up by the detail parameter
        and rounds to the closest integer. These integers then represent class indices.
        Then the class occurrences are counted and their probabilities calculated.
        :param targets: Samples of the environment P
        :param predictions: Samples of the model Q
        :param detail: The detail with which we compare. Determines the amount of classes. Can be any positive real.
        :return: Discretize probability distributions p and q
        '''

        # Normalize each dimension across all samples.
        try:
            max = torch.stack([predictions, targets]).max(0)[0][0]
        except RuntimeError:
            print(predictions)
            print(targets)
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