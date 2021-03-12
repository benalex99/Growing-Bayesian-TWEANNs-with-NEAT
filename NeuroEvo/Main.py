<<<<<<< Updated upstream
from NeuroEvo.Environments.GymEnv import GymEnv
from NeuroEvo.Optimizers.NEAT.NEAT import *
from NeuroEvo.Optimizers.Trainer import Trainer

optim = NEAT(iterations= 1000000000000, batchSize= 100, maxPopSize= 2, episodeDur= 400, showProgress= (1,1000))
# env = GymEnv('CartPole-v0')
# env = GymEnv('MountainCar-v0')
env = GymEnv('LunarLander-v2')
# # env = GymEnv('LunarLanderContinuous-v2')
# # env = GymEnv('Acrobot-v1')
# # env = GymEnv('Pendulum-v0')
# # env = GymEnv('MountainCarContinuous-v0')
# # env = GymEnv('BipedalWalker-v3')
# # env = GymEnv('Copy-v0')
#
# print(env.outputs())
gg, score = Trainer.run(optim,env)
#
gg.visualize(ion= False)
# print(gg.nodes)
# print(gg.edges)
# Testing.Testing.test()
=======
import Testing
from Optimizers.NEAT.NEAT import NEAT
from Optimizers.Trainer import Trainer
from Environments.GymEnv import GymEnv
from Optimizers.QLearner.QLearner import QPolicy
import gym
from Environments.Classification import BayesianClassification

def neatTest():
    optim = NEAT(iterations= 1000000000000, batchSize= 200, maxPopSize= 100, episodeDur= 400, showProgress= (1,1000))
    # env = GymEnv('CartPole-v0')
    # env = GymEnv('MountainCar-v0')
    env = GymEnv('LunarLander-v2')
    # # env = GymEnv('LunarLanderContinuous-v2')
    # # env = GymEnv('Acrobot-v1')
    # # env = GymEnv('Pendulum-v0')
    # # env = GymEnv('MountainCarContinuous-v0')
    # # env = GymEnv('BipedalWalker-v3')
    # # env = GymEnv('Copy-v0')
    #
    # print(env.outputs())
    gg, score = Trainer.run(optim,env)
    #
    gg.visualize(ion= False)

def otherStuff():
    # Testing.Testing.test()
    # BayesianClassification.noInput(5, 1000000)

    qLearning = QPolicy('LunarLander-v2')
    qLearning.run(100000)

otherStuff()

>>>>>>> Stashed changes
