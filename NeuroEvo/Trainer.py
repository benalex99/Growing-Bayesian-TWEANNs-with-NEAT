import gym




# Trains and tests networks on the gym environments
class Trainer:

    # Run training and testing
    @staticmethod
    def run(optimizer, envs):
        nns = Trainer.train(optimizer, envs)
        Trainer.test(nns, envs)
        return nns

    # Train networks using the optimizer
    @staticmethod
    def train(optimizer, envs):
        return

    # Test networks, do some benchmarking
    @staticmethod
    def test(nns, envs):
        return