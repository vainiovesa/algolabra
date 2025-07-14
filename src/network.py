import numpy as np

class Network:
    def __init__(self, layers):
        self.weights = []
        self.biases = []

        n = len(layers)
        for i in range(n - 1):
            weights = glorot()
            self.weights.append(weights)

            biases = np.zeros()
            self.biases.append(biases)

    def feed_forward(x):
        pass

def glorot(n, m):
    pass

def sigmoid(z):
    pass
