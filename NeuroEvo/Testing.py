import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import NeuralNetwork.AbsoluteGrad.Linear as Linear
from CustomEnvs import Envi
from NeuralNetwork.ExpandingEnsemble.Ensemble import Ensemble

class Testing:
    @staticmethod
    def testWithAnalytic():
        random.seed(0)
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H, D_out = 64, 2, 3, 1

        # Create random Tensors to hold inputs and outputs
        #x = torch.randn(N, D_in)
        #y = torch.randn(N, D_out)
        xData, yData = Envi.XOR(N,2,1)

        # Construct our model by instantiating the class defined above
        model = Linear.GrowingLinear(D_in, D_out)

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the two
        # nn.Linear modules which are members of the model.
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        gradSum = 0
        sqSum = 0
        weights1 = []
        weights2 = []
        grads = []
        iter = 10000
        testIter = 20

        for t in range(iter):
            # # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(xData)
            y_pred = F.sigmoid(y_pred)
            # print("Input: " + str(x[0]) + " Output: " + str(y_pred[0].item()))
            # print(model.lin1.weight)
            # print(model.lin2.weight)

            # Compute and print loss
            loss = criterion(y_pred, yData)
            # print(loss)
            if t % 100 == 99:
                print("Iteration: " + str(t) + " Loss: " + str(loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gradSum += model.linHidden.weight.detach().numpy().copy()
            sqSum += pow(model.linHidden.weight.detach().numpy().copy(), 2)

            weights1.append(model.linIn.weight.detach().numpy().copy())
            weights2.append(model.linHidden.weight.detach().numpy().copy())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        xData, yData = Envi.XOR(N,2,2)
        for t in range(testIter):

            # # Forward pass: Compute predicted y by passing x to the model
            index = np.random.randint(0,len(xData))
            x = xData[t]
            y = yData[t]

            y_pred = model(x)
            y_pred = F.sigmoid(y_pred)
            # print("Input: " + str(x[0]) + " Output: " + str(y_pred[0].item()))
            # print(model.lin1.weight)
            # print(model.lin2.weight)

            # Compute and print loss
            loss = criterion(y_pred, y)
            # print(loss)
            if t % 100 == 99:
                print("Iteration: " + str(t) + " Loss: " + str(loss.item()))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            gradSum += model.linHidden.weight.detach().numpy().copy()
            sqSum += pow(model.linHidden.weight.detach().numpy().copy(), 2)

            weights1.append(model.linIn.weight.detach().numpy().copy())
            weights2.append(model.linHidden.weight.detach().numpy().copy())
        weights1.append(model.linIn.weight.detach().numpy().copy())
        weights2.append(model.linHidden.weight.detach().numpy().copy())
        iter +=testIter
        print([0,0], round(model(torch.tensor([0,0]).float()).item()) == 0)
        print([0,1], round(model(torch.tensor([0,1]).float()).item()) == 1)
        print([1,0], round(model(torch.tensor([1,0]).float()).item()) == 1)
        print([1,1], round(model(torch.tensor([1,1]).float()).item()) == 0)

        weights1 = np.array(weights1).reshape((-1, iter+1))
        weights2 = np.array(weights2).reshape((-1, iter+1))

        grads = np.array(grads)

        fig, ax = plt.subplots(len(weights1) + len(weights2), 2)

        weights1 = weights1[:,len(weights1[0])-20:]
        weights2 = weights2[:,len(weights2[0])-20:]

        xValues = range(0,len(weights1[0]))

        ax[0,0].set_title("L1")
        for i, w in enumerate(weights1):
            ax[i, 0].plot(xValues, w)
            ax[i, 0].plot(xValues, w, "bo")
            ax[i, 0].set_ylabel("w" + str(i))

        ax[0,1].set_title("L2")
        for i, w in enumerate(weights2):
            ax[i, 1].plot(xValues, w)
            ax[i, 1].plot(xValues, w, "bo")
            ax[i, 1].set_ylabel("w" + str(i))

        plt.show()

    @staticmethod
    def test():
        D_in, D_out, H = 2, 1, 2
        xData, yData = Envi.XOR(500, D_in)

        model = Ensemble(D_in, D_out, H)

        model.Train(xData, yData)

        print("Average Loss: " + str((yData - model(xData)).abs().mean().item()))
        print("Module count: " + str(len(model.nns)))
        model.info()

