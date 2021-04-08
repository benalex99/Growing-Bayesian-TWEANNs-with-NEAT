import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import pyro
from pyro.optim import Adam
from pyro.distributions import *
from pyro.infer import Trace_ELBO, SVI
from tqdm import tqdm


def data():
    data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),
                      MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),
                      MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)).sample([50]),
                      MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50])))
    return data

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

        # Sample cluster probabilities between 0 and 1 for each cluster
        with pyro.plate("beta_plate", self.T-1):
            beta = pyro.sample("beta", Beta(1, alpha))

        # Sample cluster means from a multivariate normal with means 0,0 and variances 5,5
        with pyro.plate("mu_plate", self.T):
            mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))

        # Generate cluster assignments based on their weights
        # Sample data points from the assigned clusters
        with pyro.plate("data", self.N):
            z = pyro.sample("z", Categorical(self.mix_weights(beta)))
            print(MultivariateNormal(mu[z], torch.eye(2)).sample([1]).shape)
            pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)


    def guide(self, data):
        """
        The inference guide that declares our parameters, their use and their priors.
        :param data: Our data. It is not directly used, but guides the sampling.
        :return:
        """
        T = self.T
        N = len(data)

        # Cluster probability prior from uniform
        kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
        # Cluster mean prior from multivariate
        tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)).sample([T]))
        # Cluster assignment probability prior from dirichlet
        phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)

        # Sample cluster probabilities, for each cluster - 1
        with pyro.plate("beta_plate", T-1):
            q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))

        # Sample cluster means
        with pyro.plate("mu_plate", T):
            q_mu = pyro.sample("mu", MultivariateNormal(tau, torch.eye(2)))

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

    def run(self):
        self.data = data()
        print("shape " +str(self.data.shape))
        self.T = 6
        optim = Adam({"lr": 0.05})
        self.svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())
        self.losses = []

        pyro.set_rng_seed(0)
        self.alpha = 0.4
        truncationFactor = 0.1
        self.train(1000)

        # We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
        Bayes_Centers_01, Bayes_Weights_01 = self.truncate(truncationFactor, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

        pyro.set_rng_seed(0)
        self.alpha = 1.5
        truncationFactor = 0.1
        self.train(1000)

        # We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
        Bayes_Centers_15, Bayes_Weights_15 = self.truncate(truncationFactor, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(self.data[:, 0], self.data[:, 1], color="black")
        plt.scatter(Bayes_Centers_01[:, 0], Bayes_Centers_01[:, 1], color="red")
        for center, weight in zip(Bayes_Centers_01, Bayes_Weights_01):
            plt.annotate(str(weight.item()), center, color="red")

        plt.subplot(1, 2, 2)
        plt.scatter(self.data[:, 0], self.data[:, 1], color="black")
        plt.scatter(Bayes_Centers_15[:, 0], Bayes_Centers_15[:, 1], color="red")
        for center, weight in zip(Bayes_Centers_15, Bayes_Weights_15):
            plt.annotate(str(weight.item()), center, color="red")
        plt.tight_layout()
        plt.show()