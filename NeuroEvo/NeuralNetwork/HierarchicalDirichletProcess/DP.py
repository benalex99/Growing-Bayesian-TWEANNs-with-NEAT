import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pyro
from pyro.distributions import *


class DP():
    assert pyro.__version__.startswith('1.6.0')
    pyro.set_rng_seed(0)

    def betaDraws(self, a, n):
        return Beta(1,a).sample((n,1))

    def mix_weights(self, beta):
        beta1m_cumprod = (1 - beta).cumprod(-1)
        return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

    def test(self):
        self.test2()

    def test2(self):
        data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),
                          MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),
                          MultivariateNormal(torch.tensor([1.5, 2]), torch.eye(2)).sample([50]),
                          MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50])))

        plt.scatter(data[:, 0], data[:, 1])
        plt.title("Data Samples from Mixture of 4 Gaussians")
        plt.show()
        N = data.shape[0]

        draws = self.betaDraws(5, N)
        self.model(draws)

    def model(self, data):
        alpha = 0.4
        T = clusterCount = 10
        with pyro.plate("beta_plate", T-1):
            beta = pyro.sample("beta", Beta(1, alpha))

        with pyro.plate("mu_plate", T):
            mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), 5 * torch.eye(2)))

        with pyro.plate("data", N):
            z = pyro.sample("z", Categorical(self.mix_weights(beta)))
            pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)

    def test999(self):
        data = np.random.multinomial(1, [0.75,0.25], 100)
        data2 = np.random.multinomial(1, [0.25,0.75], 100)
        alpha = [1,1]
        print((data.sum(0) + data2.sum(0))/2 )

        for i in range(50):
            pred = 0
            for i in range(10):
                alpha += data.sum(0)
                pred = np.random.dirichlet(alpha, 1)

            for i in range(10):
                alpha += data2.sum(0)
                pred = np.random.dirichlet(alpha, 1)
            print(pred)
