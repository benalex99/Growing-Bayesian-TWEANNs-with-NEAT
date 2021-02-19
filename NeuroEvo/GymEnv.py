import gym
from gym.spaces import Box, Discrete





class GymEnv:
    def __init__(self, envName):
        self.env = gym.make(envName)

    def test(self, population):
        for genome in population:
            observation = self.env.reset()
            nn = genome.toNN()
            for _ in range(1000):
                if(self.env.done):
                    break
                action = self.getAction(nn, observation)
                observation, reward, done, info = self.env.step(action)

    def inputs(self):
        if(isinstance(self.env.observation_space, Discrete)):
            return self.env.observation_space.n
        else:
            return len(self.env.observation_space.high)

    def outputs(self):
        if (isinstance(self.env.action_space, Discrete)):
            return self.env.action_space.n
        else:
            return len(self.env.action_space.high)

    def getAction(self, nn, observation):
        if(isinstance(self.env.action_space, Discrete)):
            outputs = nn.forward(observation)
            max = 0
            for i, output in enumerate(outputs):
                if (output > max):
                    max = output
                    action = i
        else:
            action = nn.forward(observation)

        return action

