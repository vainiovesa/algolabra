import numpy as np
from math import sqrt

class Network:
    def __init__(self, layers):
        self.weights = []
        self.biases = []

        n = len(layers)
        for i in range(1, n):
            weights = glorot(layers[i - 1], layers[i])
            self.weights.append(weights)

            biases = np.zeros(layers[i])
            self.biases.append(biases)

    def feed_forward(x):
        pass

def glorot(n, m):
    """Weight initialization function suitable for the sigmoid activation function

    Args:
        n (int): Number of inputs for this layer ("fan-in")
        m (int): Number of outputs (neurons) for this layer ("fan_out")

    Returns:
        np.ndarray: Weights for this layer
    """
    b = sqrt(6 / (n + m))
    a = - b
    return np.random.uniform(a, b, (m, n))

def sigmoid(z):
    pass
