import matplotlib.pyplot as plt
import random

class BayesianClassification:

    @staticmethod
    def noInput(nrOfObjects, sampleSize):
        sample = []
        for i in range(sampleSize):
            sample.append(random.randint(0,nrOfObjects))

        plt.hist(sample)
        plt.show()
