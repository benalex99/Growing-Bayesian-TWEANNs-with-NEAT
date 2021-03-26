import torch
import torch.nn as nn
import random
import gym
import numpy as np

class Model(nn.Module):
    def __init__(self, layers, device = "cpu"):
        super(Model, self).__init__()
        self.device = device
        self.layers = []
        for h in layers:
            self.layers.append(nn.Linear(h[0], h[1]).to(torch.device(self.device)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        x = torch.tensor(x).float().to(torch.device(self.device))
        for i, layer in enumerate(self.layers):
            if(i < len(self.layers) - 1):
                x = torch.relu(layer(x))
            else:
                x = layer(x)
        return x.cpu()


class QPolicy():
    def __init__(self, env, sampleSize = 2000, expBufferSize = 2000, epBatchSize = 10, exploration = 0.5, lr = 0.1, discount = 0.99, seed = 0):
        self.env = []
        for i in range(epBatchSize):
            self.env.append(gym.make(env))
            self.env[i].seed(seed)
        self.testEnv = gym.make(env)

        self.input, self.output = len(self.env[0].observation_space.high), self.env[0].action_space.n
        self.model = Model([(self.input, 1000),(1000,1000),(1000,500),(500,500),(500,500),(500,self.output)], device="cuda")
        self.seed = seed
        self.random = random
        self.random.seed(seed)

        self.sampleSize = sampleSize
        self.expBufferSize = expBufferSize
        self.epBatchSize = epBatchSize
        self.explorationSetter = exploration
        self.lr = lr
        self.discount = discount

    def run(self, iter):
        self.expBuffer = []
        self.exploration = self.explorationSetter

        for i in range(iter):

            # Simulate new experience
            observations, actions, rewards, actionValues = self.episode()
            # Store experience
            # self.manageExpBuffer(observationBatches, actionsBatches, rewardsBatches, actionValuesBatches)
            # Sample experience for learning
            # observations, actions, rewards, actionValues = self.sampleBuffer(self.sampleSize)
            # Update the experience with current policy knowledge
            observations, actionValues = self.updateExperience(observations, actions, rewards, actionValues)
            # Train neural network with updated experiences
            self.trainNN(observations, actionValues)

            self.exploration -= self.explorationSetter / iter

            print("Episode: " + str(i) + "  Runs: " + str(i*self.epBatchSize))
            if i%10 == 0:
                self.test()

    def episode(self):
        observationBatch = []
        done = []
        for env in self.env:
            observationBatch.append(env.reset())
            done.append(False)

        steps = 0

        observations = []
        actions = []
        rewards = []
        actionVals = []
        # Keep going until trial has ended
        while not all(done):
            steps += 1

            # Sample actions
            if random.randrange(0,1) <= self.exploration:
                valueBatch, actionBatch, actionValuesBatch = self.explore(observationBatch)
            else:
                valueBatch, actionBatch, actionValuesBatch = self.exploit(observationBatch)

            observations.append(observationBatch)
            actions.append(actionBatch)

            # Advance environment by one step
            observationBatch = []
            rewardBatch = []
            for i, env in enumerate(self.env):
                observation, reward, done[i], _ = env.step(actionBatch[i])
                observationBatch.append(observation)
                rewardBatch.append(reward)

            rewards.append(rewardBatch)
            actionVals.append(actionValuesBatch)

        print("steps :" + str(steps))
        return np.array(observations).reshape((-1, self.input)), \
               np.array(actions).flatten(), \
               np.array(rewards).flatten(), \
               np.array(actionVals).reshape((-1, self.output))

    # Return a random action
    def explore(self, obsBatch):
        actionValuesBatch = self.model(obsBatch).detach()
        valuesBatch = []
        actionBatch = []
        for actionValue in actionValuesBatch:
            action = self.random.randint(0, self.output - 1)
            actionBatch.append(action)
            valuesBatch.append(actionValue[action].item())

        return valuesBatch, actionBatch, actionValuesBatch.numpy()

    # Return the best action
    def exploit(self, obs):
        actionValues = self.model(obs).detach()
        if np.array(obs).ndim > 1:
            value, action = torch.max(actionValues, 1)
        else:
            value, action = torch.max(actionValues, 0)

        return value, action, actionValues.numpy()

    # Update the network with the new policy
    def trainNN(self, x, y, iter = 50, lr=0.0002):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr)

        avgLoss = 0
        t = 0
        while(t < iter):
            t += 1
            # # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(x)

            # Compute and print loss
            loss = criterion(y_pred, torch.tensor(y))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lastloss= loss.detach().item()
            avgLoss += lastloss
        print("Loss :" + str(lastloss))

    def test(self):

        observation = self.testEnv.reset()

        done = False
        steps = 0
        actionSum = 0
        rewardSum = 0
        while not done:
            steps += 1
            self.testEnv.render()

            value, action, actionValues = self.exploit(observation)
            actionSum += actionValues
            # Advance environments by one step
            observation, reward, done, info = self.testEnv.step(action.item())

            rewardSum+= reward
        print("Money: " + str(rewardSum))
        print("Moves: " + str(actionSum/steps))

    # Performs Q-learning temporal propagation and returns the observations + updated predictions
    def updateExperience(self, observations, actions, rewards, actionValues):
        newValBatch, _, _ = self.exploit(observations)
        for newVal, act, rew, actVals in zip(newValBatch, actions, rewards, actionValues):
            actVals[act] = actVals[act] + self.lr * ((rew + self.discount * newVal) - actVals[act])

        return observations, actionValues

    def manageExpBuffer(self, observationBatches, actionBatches, rewardBatches, actionValueBatches):
        for observations, actions, rewards, actionValues in zip(observations, actions, rewards, actionValues):
            self.expBuffer.append([observations, actions, rewards, actionValues])
        while(len(self.expBuffer) > self.expBufferSize):
            self.expBuffer.pop(0)

    def sampleBuffer(self,N):
        sample = []
        for _ in range(min(N,len(self.expBuffer))):
            sample.append(self.expBuffer[random.randint(0, len(self.expBuffer) - 1)])

        sample = np.array(sample)

        # observationBatches, actionBatches, rewardBatches, actionValueBatches
        return list(sample[:,0]), list(sample[:,1]), list(sample[:,2]), list(sample[:,3]),