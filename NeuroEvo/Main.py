from NeuroEvo.Optimizers.NEAT.NEAT import *
from NeuroEvo.GymEnv import GymEnv
from NeuroEvo.Trainer import Trainer
import gym
import matplotlib as plt
import torch


optim = NEAT(iterations= 1000000000000, batchSize= 200, maxPopSize= 400, episodeDur= 1000, showProgress= (1,1000))
#env = GymEnv('CartPole-v0')
#env = GymEnv('MountainCar-v0')
env = GymEnv('LunarLander-v2')
#env = GymEnv('LunarLanderContinuous-v2')
#env = GymEnv('Acrobot-v1')
#env = GymEnv('Pendulum-v0')
#env = GymEnv('MountainCarContinuous-v0')

print(env.inputs())
print(env.outputs())

gg, score = Trainer.run(optim,env)

gg.visualize(ion= False)
print(gg.nodes)
print(gg.edges)

# a = torch.tensor([1,2,3], dtype=torch.float)
# b = torch.tensor([[10,100,1000],
#                   [1,0.1,0.01]], dtype= torch.float)
#
# print(a)
# print(b)
# c = torch.matmul(a,b.t())
# print(c)
# d = a * b
# print(d)
# e = d.prod(axis=1)
# print(e)