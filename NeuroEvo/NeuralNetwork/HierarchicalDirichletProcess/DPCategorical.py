import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import pyro
from pyro.optim import Adam
from pyro.distributions import *
from pyro.infer import Trace_ELBO, SVI
from tqdm import tqdm


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
        The model with static parameters
        :param data: The data whose probability we attempt to find.
        :return: Nonesies
        """

        # alpha = 0.4
        alpha = self.alpha
        self.N = len(data)

        # Sample cluster probabilities between 0 and 1 for each cluster, for the stick breaking process
        with pyro.plate("beta_plate", self.T-1):
            beta = pyro.sample("beta", Beta(1, alpha))

        # Sample categorical probabilities from a dirichlet with uniform class counts
        with pyro.plate("mu_plate", self.T):
            mu = pyro.sample("mu", Dirichlet(1/self.classCount * torch.ones(self.classCount)))

        with pyro.plate("data", self.N):
            # Generate cluster assignments based on their weights from the stick breaking process
            z = pyro.sample("z", Categorical(self.mix_weights(beta)))
            # Sample data points from the assigned clusters
            pyro.sample("obs", Categorical(mu[z]), obs=data)

    def guide(self, data):
        """
        The inference guide that declares our parameters, their use and their priors.
        :param data: Our data. It is not directly used, but guides the sampling.
        :return:
        """
        T = self.T
        N = len(data)

        # Stick breaking parameter prior from uniform
        kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
        # Cluster dirichlet pseudocount from dirichlet-multinomial. 1000 is an arbitrary choice
        tau = pyro.param('tau', lambda: DirichletMultinomial(torch.ones(self.classCount), 1000).sample([T]), constraint=constraints.positive)
        # Cluster assignment probability prior from dirichlet
        phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

        # Sample cluster probabilities, for each cluster - 1
        with pyro.plate("beta_plate", T-1):
            q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

        # Sample cluster parameters/ categorical distributions
        with pyro.plate("mu_plate", T):
            q_mu = pyro.sample("mu", Dirichlet(tau))

        # Sample cluster assignments
        with pyro.plate("data", N):
            z = pyro.sample("z", Categorical(phi))

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

    def data(self, N= 200, classCount = 10):
        self.data1 = Categorical(Dirichlet(1/classCount * torch.ones(classCount)).sample([1])).sample([N])
        self.data2 = Categorical(Dirichlet(1/classCount * torch.ones(classCount)).sample([1])).sample([N])
        data = torch.cat((self.data1, self.data2)).flatten()
        return data

    def run(self, classCount= 10):
        self.data = self.data(classCount=classCount)
        self.classCount = classCount

        self.T = 6
        optim = Adam({"lr": 0.05})
        self.svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        self.losses = []

        pyro.set_rng_seed(0)
        self.alpha = 1.5
        truncationFactor = 0.1
        self.train(1000)

        # We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
        categoricals, weights = self.truncate(truncationFactor, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

        distr1 = torch.ones(10)
        distr2 = torch.ones(10)
        distr = torch.ones(10)
        for i in self.data1:
            distr1[i] += 1
        for i in self.data2:
            distr2[i] += 1
        for i in self.data:
            distr[i] += 1

        print(Dirichlet(distr2).entropy())
        print(Dirichlet(distr1).entropy()/2+Dirichlet(distr2).entropy()/2)
        print(Dirichlet(distr).entropy())

        plt.figure(figsize=(15, 5))
        plt.subplot(3, 3, 1)
        plt.bar(range(10), distr1, color="blue")
        plt.subplot(3, 3, 2)
        plt.bar(range(10), distr2, color="blue")
        plt.subplot(3, 3, 3)
        plt.bar(range(10), distr, color="blue")

        for i, categorical in enumerate(categoricals):
            plt.subplot(3, 3, 4+i)
            plt.bar(range(10), categorical, color="red")
        plt.tight_layout()
        plt.show()