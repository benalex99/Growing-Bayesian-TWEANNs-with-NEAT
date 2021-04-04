import torch
import torch.nn as nn
import copy
from NeuroEvo.NeuralNetwork.UnclassifiedExperiments.SomeLayerModels import XOR as XOR
import math

class DeepRelu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DeepRelu, self).__init__()
        self.lin1 = nn.Linear(D_in, H)
        self.lin2 = nn.Linear(H, H)
        self.lin3 = nn.Linear(H, H)
        self.lin4 = nn.Linear(H, D_out)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(torch.relu(x))
        x = self.lin3(torch.relu(x))
        x = self.lin4(torch.relu(x))
        return torch.sigmoid(x)

class ShallowRelu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ShallowRelu, self).__init__()
        self.lin1 = nn.Linear(D_in, H)
        self.lin2 = nn.Linear(H,D_out)

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin2(torch.relu(x))
        return x

class UnableRelu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(UnableRelu, self).__init__()
        self.lin1 = nn.Linear(D_in, D_out)

    def forward(self, x):
        x = self.lin1(x)
        return x

class EnsembleModule(nn.Module):
    def __init__(self, model):
        super(EnsembleModule, self).__init__()
        self.model = model
        self.dataPointsLearned = 0

    def forward(self, x):
        return self.model(x)

    def Train(self, x, y, criterion, optimizer, iter=100):
        optimizer = optimizer(self.parameters(), lr=0.01)

        y_pred = self(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totalLoss = loss.abs().sum()

        for _ in range(iter):
            y_pred = self(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.abs().sum()

        return totalLoss

# We discard old hypothesis that havent been used.
# If we have too much change to our current hypothesis, we may create a separate hypothesis to explain away the new data.
# We may not actually fully think through how we would have to change to adapt our old hypothesis to fit the new datapoint.
# Hence we also only run a few iterations of gradient descent to determine the difference.
# Currently we use the weight difference/gradient as approximation for change.
# Really we should be using a sample of our previous hypothesis explained datapoints
# and compare the outcomes on the same inputs for the new and old hypothesis. Based on the improvement or
# worsening we can then choose to adopt this changed hypothesis, or to create a separate one.
class Ensemble(nn.Module):
    def __init__(self, D_in, D_out, H = 10, murderPercentage = 0.001, growthThreshold = 40, expertPercentage = 0.6):
        super(Ensemble, self).__init__()
        self.D_in = D_in
        self.H = H
        self.D_out = D_out
        self.murderPercentage = murderPercentage
        self.growthThreshold = growthThreshold
        self.expertPercentage = expertPercentage
        self.nns = []
        self.moduleType = ShallowRelu
        self.nns.append(EnsembleModule(self.moduleType(D_in, H, D_out)))
        self.totalDataPoints = 1

    def forward(self, x):
        output = torch.zeros(len(x), self.D_out)
        totalDataPoints = 0
        for module in self.nns:
            output += module(x) * module.dataPointsLearned
            totalDataPoints += module.dataPointsLearned
        output = output / totalDataPoints

        return output

    def Train(self, xData, yData, criterion = torch.nn.MSELoss(), optimizer = torch.optim.Adam):
        print("training")
        # For each singular data point in the data
        for i, (x, y) in enumerate(zip(xData, yData)):
            moduleCopies = []
            totalGradients = []
            # Test all modules on learning the new data point
            for module in self.nns:
                moduleCopy = copy.deepcopy(module)
                moduleCopies.append(moduleCopy)
                totalGradients.append(moduleCopy.Train(x, y, criterion, optimizer, iter = 1)
                                      * moduleCopy.dataPointsLearned)

            # Update the module with the least total loss
            # Use model probability/age and model accuracy?
            # Gradient will grow with amount of data points... good or bad? Normalize back down? But its sensible scaling.
            for _ in range(max(1, round(len(self.nns)*self.expertPercentage))):
                bestIndex = totalGradients.index(min(totalGradients))

                if(totalGradients[bestIndex] < self.growthThreshold):
                    self.nns[bestIndex] = moduleCopies[bestIndex]
                    self.nns[bestIndex].Train(x, y, criterion, optimizer, iter = 99)
                    self.nns[bestIndex].dataPointsLearned += 1
                    self.totalDataPoints += 1
                else:
                    newModule = EnsembleModule(self.moduleType(self.D_in, self.H, self.D_out))
                    self.nns.append(newModule)
                    newModule.Train(x, y, criterion, optimizer)
                    newModule.dataPointsLearned += 1
                    print("Expanded")
                    break
                totalGradients[bestIndex] = math.inf


            self.filter()

            print(str(i) + "th datapoint")

    def filter(self):
        aList = []
        for module in self.nns:
            if(module.dataPointsLearned / self.totalDataPoints > self.murderPercentage):
                aList.append(module)
            else:
                print("Murder")
        self.nns = aList

    def info(self):
        for module in self.nns:
            print("Weights: " + str(module.dataPointsLearned))
            print(module.model.lin1.weight.data)
            print(module.model.lin1.bias.data)
            print(module.model.lin2.weight.data)
            print(module.model.lin2.bias.data)
        print([0,0], self(torch.tensor([[0,0]]).float()).item())
        print([0,1], self(torch.tensor([[0,1]]).float()).item())
        print([1,0], self(torch.tensor([[1,0]]).float()).item())
        print([1,1], self(torch.tensor([[1,1]]).float()).item())

    #Simply assigning the datapoint to the network with the least error does not produce any generalized network,