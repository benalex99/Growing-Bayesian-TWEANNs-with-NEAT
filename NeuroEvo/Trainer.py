from NeuroEvo import GymEnv
from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome


# Trains and tests networks on the gym environments
class Trainer:

    # Run training and testing
    @staticmethod
    def run(optimizer, env):
        nn = Trainer.train(optimizer, env)
        score = Trainer.test(nn, env)
        return nn, score

    # Train networks using the optimizer
    @staticmethod
    def train(optimizer, env):
        env: GymEnv
        rootGenome = NEATGenome(env.inputs(), env.outputs())
        return optimizer.run(rootGenome, env)

    # Test networks, do some benchmarking
    @staticmethod
    def test(nn, env):
        env: GymEnv
        return env.test([nn])[0]
