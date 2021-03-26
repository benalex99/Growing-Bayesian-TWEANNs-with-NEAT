from NeuroEvo.Environments.GymEnv import GymEnv
from Optimizers.NEAT.NEAT import NEAT
from Optimizers.QLearner.QLearner import QPolicy
from Optimizers.Trainer import Trainer


def neatTest():
    optim = NEAT(iterations= 1000000000000, batchSize= 200, maxPopSize= 2, episodeDur= 400, showProgress= (1,1000))
    # env = GymEnv('CartPole-v0')
    # env = GymEnv('MountainCar-v0')
    # env = GymEnv('LunarLander-v2')
    # # env = GymEnv('LunarLanderContinuous-v2')
    env = GymEnv('BipedalWalker-v3')
    # # env = GymEnv('Acrobot-v1')
    # env = GymEnv('CarRacing-v0')
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

neatTest()
