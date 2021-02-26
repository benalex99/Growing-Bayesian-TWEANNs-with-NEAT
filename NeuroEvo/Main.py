from NeuroEvo.Optimizers.NEAT.NEAT import *
from NeuroEvo.GymEnv import GymEnv
from NeuroEvo.Trainer import Trainer
import gym
import matplotlib as plt
import torch


optim = NEAT(iterations= 1000000000000, batchSize= 200, maxPopSize= 400, episodeDur= 1000, showProgress= (1,1000))
env = GymEnv('CartPole-v0')
# env = GymEnv('MountainCar-v0')
#env = GymEnv('LunarLander-v2')
# env = GymEnv('LunarLanderContinuous-v2')
# env = GymEnv('Acrobot-v1')
# env = GymEnv('Pendulum-v0')
# env = GymEnv('MountainCarContinuous-v0')

gg, score = Trainer.run(optim,env)

gg.visualize(ion= False)
print(gg.nodes)
print(gg.edges)
