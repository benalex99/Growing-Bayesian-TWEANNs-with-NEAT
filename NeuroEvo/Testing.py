import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Model, self).__init__()
        self.lin2 = nn.Linear(D_in, H)
        self.xor1 = XOR(H, H)
        self.lin1 = nn.Linear(H, D_out)

    def forward(self, x):
        # x = self.lin1(x)
        # print("x1: " + str(x))
        x = self.xor1(x)
        # print("x2: " + str(x))
        return torch.sigmoid(self.lin1(x))

class XOR(nn.Module):
    def __init__(self, D_in, D_out):
        super(XOR, self).__init__()
        self.lin1 = nn.Linear(D_in, D_out)
        weights = []
        for inp in range(D_in):
            w = np.ones(D_out, dtype= float) * -1
            w[inp] = 1
            weights.append(w)
        with torch.no_grad():
            self.lin1.weight = nn.Parameter(torch.tensor(weights, dtype= torch.float))
        self.lin1.requires_grad_(False)

        self.weight = self.lin1.weight

    def forward(self, x):
        return F.relu(self.lin1(x))

class NonMaxUnlearn(nn.Module):
    def __init__(self, D_in, D_out):
        super(NonMaxUnlearn, self).__init__()
        self.xor = XOR(D_in, D_out)

    def forward(self, x):
        output, _ = torch.max(torch.abs(x), 1)

        maxes, _ = torch.max(self.xor(x), 1)
        mins, _ = torch.min(self.xor(x), 1)

        maxSigns = torch.sign(maxes)
        minSigns = torch.sign(mins)

        signs = maxSigns.int() | minSigns.int()
        result = output * signs
        return torch.reshape(result, (len(result),1))

class Envi():
    @staticmethod
    def sample(n):
        input = []
        output = []
        for i in range(0, n):
            r = random.randint(0,8)
            if r == 0:
                input.append([0.0, 0.0, 0.0])
                output.append([0.0])
            elif r == 1:
                input.append([1.0,0.0,0.0])
                output.append([1.0])
            elif r == 2:
                input.append([0.0,1.0,0.0])
                output.append([1.0])
            elif r == 3:
                input.append([1.0,1.0,0.0])
                output.append([0.0])
            elif r == 4:
                input.append([0.0,1.0,1.0])
                output.append([0.0])
            elif r == 5:
                input.append([1.0,0.0,1.0])
                output.append([0.0])
            elif r == 6:
                input.append([0.0,1.0,1.0])
                output.append([0.0])
            elif r == 7:
                input.append([0.0,0.0,1.0])
                output.append([1.0])
            else:
                input.append([1.0,1.0,1.0])
                output.append([0.0])


        return torch.tensor(input).float(), torch.tensor(output).float()

class Testing:
    @staticmethod
    def test():
        random.seed(0)
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H, D_out = 64, 3, 3, 1

        # Create random Tensors to hold inputs and outputs
        #x = torch.randn(N, D_in)
        #y = torch.randn(N, D_out)
        x, y = Envi.sample(N)

        # Construct our model by instantiating the class defined above
        model = Model(D_in, H, D_out)

        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the two
        # nn.Linear modules which are members of the model.
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        gradSum = 0
        sqSum = 0
        weights = []
        grads = []
        iter = 50000
        print("Weight: " + str(model.lin1.weight.detach().numpy()) )
        for t in range(iter):
            # # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x)
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
            gradSum += model.lin1.weight.detach().numpy().copy()
            sqSum += pow(model.lin1.weight.detach().numpy().copy(), 2)
            weights.append(model.lin1.weight.detach().numpy().copy())
            # grads.append(model.lin1.weight.grad.detach().numpy().copy())

        weights.append(model.lin1.weight.detach().numpy().copy())
        print("GradSum: " + str(gradSum))
        print("sqSum: " + str(sqSum))

        gradMean = gradSum / iter
        stDev = np.sqrt((sqSum / iter) - np.square(gradMean))

        print("Mean: " + str(gradMean))
        print("stDev: " + str(stDev))

        print("Weight: " + str(model.lin1.weight.detach().numpy()) )

        # Converged
        # Mean: [[-12.73863818  11.72467281]
        #  [-11.89546613  12.26476286]]
        # stDev: [[0.8903341  0.85618427]
        #  [1.10187019 1.10423262]]

        # Not Converged
        # Mean: [[ -5.1842766 -12.79832  ]
        #  [  7.3667736 -13.064345 ]]
        # stDev: [[0.54236597 2.2329712 ]
        #  [0.49488503 2.2005575 ]]

        # Not Converged 1 node
        # Mean: [[-14.340984 -14.080064]]
        # stDev: [[2.0686867 2.0696084]]

        print(round(model(torch.tensor([0,0,0]).float()).item()) == 0)
        print(round(model(torch.tensor([0,1,0]).float()).item()) == 1)
        print(round(model(torch.tensor([1,1,0]).float()).item()) == 0)
        print(round(model(torch.tensor([0,0,1]).float()).item()) == 1)
        print(round(model(torch.tensor([1,0,0]).float()).item()) == 1)
        print(round(model(torch.tensor([0,1,1]).float()).item()) == 0)
        print(round(model(torch.tensor([1,0,1]).float()).item()) == 0)
        print(round(model(torch.tensor([1,1,1]).float()).item()) == 0)

        print("Weight: " + str(model.lin1.weight.detach().numpy()) )
        weights = np.array(weights).reshape((3, iter+1))
        # print(weights[:,0,0])
        grads = np.array(grads)
        fig, ax = plt.subplots(len(weights),1)
        # ax[0] = plt.hist(weights[0,0], bins = 200)
        # ax[1] = plt.hist(weights[1,0], bins = 200)
        # ax[2] = plt.hist(weights[2,0], bins = 200)

        for i, w in enumerate(weights):
            ax[i].plot(w)
        plt.show()