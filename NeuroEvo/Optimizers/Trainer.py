from NeuroEvo.Optimizers.NEAT.NEATGenome import NEATGenome


# Trains and tests networks on the gym environments
class Trainer:

    # Run training and testing
    @staticmethod
    def run(optimizer, env):
        nn = Trainer.train(optimizer, env)
        Trainer.test(nn, env)
        return nn, nn.fitness

    # Train networks using the optimizer
    @staticmethod
    def train(optimizer, env):
        rootGenome = NEATGenome(env.inputSize(), env.outputSize())
        return optimizer.run(rootGenome, env)

    # Test networks, do some benchmarking
    @staticmethod
    def test(nn, env):
        env.finalTest(nn)
