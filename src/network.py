import numpy as np
from math import sqrt

class Network:
    def __init__(self, layers:list):
        self.n_inputs = layers[0]
        self.weights = []
        self.biases = []

        n = len(layers)
        for i in range(1, n):
            weights = glorot(layers[i - 1], layers[i])
            self.weights.append(weights)

            biases = np.zeros(layers[i])
            self.biases.append(biases)

    def feed_forward(self, x:np.ndarray):
        """Get all activations of the neural network with input x

        Args:
            x (np.ndarray): Input for the neural network

        Returns:
            list: All activations of the neural network
        """
        assert(len(x) == self.n_inputs)

        activations = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, x) + b
            a = sigmoid(z)
            activations.append(a)
            x = a
        return activations

    def evaluate(self, x:np.ndarray):
        """Get the activation of the neural network with input x

        Args:
            x (np.ndarray): Input for the neural network

        Returns:
            np.ndarray: Output layer activation of the neural network
        """
        return self.feed_forward(x)[-1]

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

def sigmoid(z:np.ndarray):
    """Activation function for the neural network to introduce nonlinearity

    Args:
        z (np.ndarray): Weighted sum

    Returns:
        np.ndarray: Vector of values between zero and one
    """
    return 1 / (1 + np.exp(-z))
