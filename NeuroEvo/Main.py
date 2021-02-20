from NeuroEvo.Optimizers.NEAT.NEAT import *
from NeuroEvo.GymEnv import GymEnv
from NeuroEvo.Trainer import Trainer
import gym
import matplotlib as plt


optim = NEAT(iterations= 50, batchSize= 20, maxPopSize= 20)
env = GymEnv('CartPole-v0')
#env = GymEnv('MountainCar-v0')
print(env.inputs())
print(env.outputs())
gg, score = Trainer.run(optim,env)

gg.visualize(ion= False)
print(gg.nodes)
print(gg.edges)