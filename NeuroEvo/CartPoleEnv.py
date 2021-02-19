import gym






class CartPoleEnv:
    def test(self, population):
        env = gym.make('CartPole-v0')

        for genome in population:
            env.reset()
            nn = genome.toNN()
            for _ in range(1000):
                if(env.done):
                    break
                action = nn.forward(env.observation_space)
                env.step(action)
