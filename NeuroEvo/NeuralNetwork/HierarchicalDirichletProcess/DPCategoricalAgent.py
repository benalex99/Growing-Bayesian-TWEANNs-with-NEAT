import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import pyro
from pyro.optim import Adam
from pyro.distributions import *
from pyro.infer import Trace_ELBO, SVI
from tqdm import tqdm
import NeuroEvo.NeuralNetwork.HierarchicalDirichletProcess.DPEnvironment as DPEnv

class DP():
    @staticmethod
    def betaDraws(a, n):
        """
        Draw some values for the cluster probabilities.
        :param a: The shape parameter a of the distribution. It determines how quickly the pdf rises with a > 1 and falls with a < 1.
        :param n: The amount of samples.
        :return: The samples.
        """

        return Beta(1, a).sample((n, 1))

    @staticmethod
    def mix_weights(beta):
        """
        Calculates the cluster probabilities based on the stick breaking process.
        :param beta: Independently sampled cluster priors
        :return: Probabilities for all clusters, which sum up to 1
        """

        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    @staticmethod
    def test():
        assert pyro.__version__.startswith('1.6.0')

        dp = DP()
        dp.run()

    def model(self, data):
        """
        The model with unattuned parameters
        :param data: The data whose probability we attempt to find.
        :return: Nonesies
        """

        # alpha = 0.4
        alpha = self.alpha
        self.N = len(data)

        # Sample T-1 many cluster probabilities between 0 and 1 for each cluster, for the stick breaking process
        with pyro.plate("beta_plate", self.T-1):
            beta = pyro.sample("beta", Beta(1, alpha))

        # Our allow the categoricals to take any distribution
        with pyro.plate("mu_plate", self.T):
            mu = pyro.sample("mu", Dirichlet(1/self.featureCount * torch.ones(self.featureCount)))

        with pyro.plate("data", self.N):
            # Generate cluster assignments based on their weights from the stick breaking process
            z = pyro.sample("z", Categorical(self.mix_weights(beta)))
            # Sample data points from the assigned clusters
            pyro.sample("obs", Categorical(mu[z]), obs=data)

    def guide(self, data):
        """
        The inference guide that declares how to explore the parameter space
        :param data: Our data. It is not directly used, but we need the shape.
        :return:
        """
        T = self.T
        N = len(data)

        # Stick breaking parameter prior from uniform
        kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)

        # Sample N many hypothetical cluster probabilities, for T many clusters, from a dirichlet
        phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

        # Sample T many hypothetical dirichlet pseudocounts from a DirichletMultinomial
        # TODO: 10000 is an arbitrary choice. Since tau+= 1 for numerical reasons,
        #  a higher number increases accuracy as it lessens the impact of tau += 1
        tau = pyro.param('tau', lambda: DirichletMultinomial(1/self.featureCount * torch.ones(self.featureCount),10000).sample([T]), constraint=constraints.positive)
        tau.data += 0.0001

        # Sample cluster probabilities, for each cluster - 1
        with pyro.plate("beta_plate", T-1):
            pyro.sample("beta", Beta(torch.ones(T-1), kappa))

        # Parameterize the dirichlet for generating our cluster distributions
        with pyro.plate("mu_plate", self.T):
             pyro.sample("mu", Dirichlet(tau))

        # Sample cluster assignments
        with pyro.plate("data", N):
            pyro.sample("z", Categorical(phi))

    def train(self, num_iterations):
        pyro.clear_param_store()
        for j in tqdm(range(num_iterations)):
            loss = self.svi.step(self.data)
            self.losses.append(loss)

    def truncate(self, alpha, centers, weights):
        threshold = alpha**-1 / 100.
        true_centers = centers[weights > threshold]
        true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
        return true_centers, true_weights

    def data(self, N= 200, classCount = 5, featureCount = 10):
        self.categoricals = []
        for i in range(classCount):
            self.categoricals.append(Categorical(Dirichlet(1/featureCount * torch.ones(featureCount)).sample([1])))

        self.classProbs = Dirichlet(1/classCount * torch.ones(classCount)).sample([1])[0]

        data = []
        for _ in range(N):
            data.append(self.categoricals[Categorical(self.classProbs).sample([1])[0]].sample([1]))
        data = torch.Tensor(data).flatten().int()
        return data

    def run(self, classCount= 5, featureCount = 10):
        self.data = self.data(classCount=classCount, featureCount=featureCount)
        self.classCount = classCount
        self.featureCount = featureCount

        self.T = 6
        optim = Adam({"lr": 0.05})
        self.svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        self.losses = []

        pyro.set_rng_seed(0)
        self.alpha = 1.5
        truncationFactor = 0.1
        self.train(1000)

        # We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
        dirichlets, weights = self.truncate(truncationFactor, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

        plt.figure(figsize=(15, 5))
        subplotWidth = max(len(self.categoricals), len(dirichlets))

        for i, (categorical, classProb) in enumerate(zip(self.categoricals,self.classProbs)):
            plt.subplot(3, subplotWidth, i+1)
            plt.title(round(classProb.item(),2))
            plt.bar(range(featureCount), (categorical.probs.numpy()*100).flatten(), color="blue")

        distr = torch.zeros(10)
        for i in self.data:
            distr[i] += 1

        plt.subplot(3, subplotWidth, subplotWidth + 1)
        plt.bar(range(featureCount), distr, color="green")

        for i, (dirichletCounts, weight) in enumerate(zip(dirichlets, weights)):
            plt.subplot(3, subplotWidth, subplotWidth*2 + i + 1)
            plt.title(round(weight.item(), 2))
            plt.bar(range(featureCount), dirichletCounts, color="red")
        plt.tight_layout()
        plt.show()