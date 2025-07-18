import numpy as np
from math import sqrt

class Network:
    def __init__(self, layers:list):
        self.n_inputs = layers[0]
        self.n_layers = len(layers)
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

    def loss(self, a:np.ndarray, a_hat:np.ndarray):
        """Quadratic loss function

        Args:
            a (np.ndarray): output
            a_hat (np.ndarray): expected output

        Returns:
            np.float64: loss value
        """
        return np.sum((a - a_hat) ** 2)

    def backward_pass(self, activations:list, a_hat:np.ndarray):
        """Get all delta values for calculating the gradient w.r.t each weight and bias

        Args:
            activations (list): Activations of the network by the feed_forward-method
            a_hat (np.ndarray): Desired output

        Returns:
            list: All delta values
        """
        delta_l = []

        a = activations[-1]
        delta_u = 2 * (a - a_hat) * a * (1 - a)
        delta_l.append(delta_u)

        for i in range(self.n_layers - 2, 0, -1):
            a = activations[i - 1]
            w = self.weights[i]
            d = delta_l[-1]
            sigmoid_derivative = a * (1 - a)
            delta_weight_sums = np.dot(w.transpose(), d)
            this_layer_delta = sigmoid_derivative * delta_weight_sums
            delta_l.append(this_layer_delta)
        delta_l.reverse()
        return delta_l

    def gradient_calculation(self, x:np.ndarray, a_hat:np.ndarray):
        """Get the gradient with respect to each weight and bias in the network

        Args:
            x (np.ndarray): Input
            a_hat (np.ndarray): Expected output

        Returns:
            tuple: Two lists; gradient w.r.t weights and biases
        """
        activations = self.feed_forward(x)
        deltas = self.backward_pass(activations, a_hat)
        weight_derivatives = []

        for delta, acts in zip(deltas, [x] + activations):
            delta = delta.reshape((len(delta), 1))
            weight_derivatives.append(delta * acts)
        return weight_derivatives, deltas

    def update_weights_and_biases(self, new_w:list, new_b:list, lr:float):
        """Update all the weights and biases of the network to descend the gradient

        Args:
            new_w (list): Weight derivatives
            new_b (list): Bias derivatives
            lr (float): Learning rate
        """
        for i in range(self.n_layers - 1):
            self.weights[i] -= lr * new_w[i]
            self.biases[i] -= lr * new_b[i]

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
