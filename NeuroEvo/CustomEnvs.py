import random
import torch

class Envi():
    @staticmethod
    def XOR(n, inputs, duplCount = 1):
        input = []
        output = []
        for i in range(0, n):
            inp = []
            sum = 0
            for i2 in range(inputs):
                r = random.randint(0,1)
                sum += r
                inp.append(r)

            for _ in range(duplCount):
                input.append(inp)
            for _ in range(duplCount):
                output.append([1] if (sum == 1) else [0])

        return torch.tensor(input).float(), torch.tensor(output).float()
