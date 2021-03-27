import random
import torch
import numpy as np

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
                output.append([1] if (sum == 1) else [0])

        return torch.tensor(input).float(), torch.tensor(output).float()

    @staticmethod
    def objectClasses(classCount, featureDimension, sampleSize):
        classes = []
        for _ in range(classCount):
            classFeatures = []
            for _ in range(featureDimension):
                classFeatures.append(random.randint(0, 1))
            classes.append(classFeatures)

        sampleX = []
        sampleY = []
        for _ in range(sampleSize):
            output = np.zeros(len(classes))
            index = random.randint(0, len(classes)-1)
            output[index] = 1
            sampleX.append(classes[index])
            sampleY.append(output)

        return torch.tensor(np.array(sampleX)).float(), torch.tensor(sampleY).float()