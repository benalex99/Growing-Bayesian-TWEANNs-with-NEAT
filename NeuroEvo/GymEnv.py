import gym
from gym.spaces import Box, Discrete

import time
import numpy as np


class GymEnv:
    def __init__(self, envName):
        self.env = gym.make(envName)

    def test(self, population):
        cumTime = 0
        for genome in population:
            observation = self.env.reset()
            nn = genome.toNN()
            cumReward = 0
            done = False

            for _ in range(10000):
                if(done):
                    break
                nTime = time.time()
                action = self.getAction(nn, observation)
                cumTime += time.time() - nTime
                observation, reward, done, info = self.env.step(action)
                cumReward += reward
            genome.fitness = cumReward
        print("Classifying took: " + str(cumTime))


    def finalTest(self, genome):
        nn = genome.toNN()
        observation = self.env.reset()
        for _ in range(1000000):
            self.env.render()
            action = self.getAction(nn, observation)
            observation, reward, done, info = self.env.step(action)

    def inputs(self):
        if(isinstance(self.env.observation_space, Discrete)):
            return self.env.observation_space.n + 1
        else:
            return len(self.env.observation_space.high) + 1

    def outputs(self):
        if (isinstance(self.env.action_space, Discrete)):
            return self.env.action_space.n
        else:
            return len(self.env.action_space.high)

    def getAction(self, nn, obs):
        observation = np.append(obs, [1])
        if(isinstance(self.env.action_space, Discrete)):
            outputs = nn.forward(observation)
            max = outputs[0]
            action = 0
            for i, output in enumerate(outputs):
                if (output > max):
                    max = output
                    action = i
        else:
            action = nn.forward(observation)

        return action

