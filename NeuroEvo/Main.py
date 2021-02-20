from NeuroEvo.Optimizers.NEAT.NEAT import *
from NeuroEvo.GymEnv import GymEnv
from NeuroEvo.Trainer import Trainer
import gym



optim = NEAT(iterations= 30, batchSize= 50, maxPopSize= 50)
env = GymEnv('CartPole-v0')
gg, score = Trainer.run(optim,env)

gg.visualize()