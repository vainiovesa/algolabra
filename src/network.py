import numpy as np
from math import sqrt
from random import shuffle

class Network:
    """Neural network class.

    Approximates arbitrary functions by learning from training data with gradient descent.
    Uses the sigmoid activation function and squared error.

    Attributes:
        n_inputs (int): Length of the array the network takes as input. 
        n_layers (int): Number of layers of the network.
        weights (list): List of np.ndarray; weight arrays.
        biases (list): List of np.ndarray; bias arrays.
    """

    def __init__(self, layers:list):
        """Class constructor for the neural network.

        Args:
            layers (list): List of integers; Lengths of the input layer, hidden layers and the output layer.
        """
        self.n_inputs = layers[0]
        self.n_layers = len(layers)
        self.weights = []
        self.biases = []

        for i in range(1, self.n_layers):
            weights = glorot(layers[i - 1], layers[i])
            self.weights.append(weights)

            biases = np.zeros(layers[i])
            self.biases.append(biases)

    def feed_forward(self, x:np.ndarray):
        """Get all activations of the neural network with input x.

        Args:
            x (np.ndarray): Input for the neural network.

        Returns:
            list: List of np.ndarray; activations of each layer of the neural network.
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
        """Get the activation of the neural network with input x.

        Args:
            x (np.ndarray): Input for the neural network.

        Returns:
            np.ndarray: Output layer activation of the neural network.
        """
        return self.feed_forward(x)[-1]

    def loss(self, a:np.ndarray, a_hat:np.ndarray):
        """Quadratic loss function.

        Args:
            a (np.ndarray): Output.
            a_hat (np.ndarray): Expected output.

        Returns:
            np.float64: Loss value.
        """
        return np.sum((a - a_hat) ** 2)

    def backward_pass(self, activations:list, a_hat:np.ndarray):
        """Get all delta values for calculating the gradient w.r.t each weight and bias.

        Args:
            activations (list): List of np.ndarray; activations of each layer of the network by the feed_forward-method.
            a_hat (np.ndarray): Desired output.

        Returns:
            list: List of np.ndarray; all delta values.
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
        """Get the gradient with respect to each weight and bias in the network.

        Args:
            x (np.ndarray): Input.
            a_hat (np.ndarray): Expected output.

        Returns:
            tuple: Two lists and np.float64; gradient w.r.t the weights and biases, loss.
        """
        activations = self.feed_forward(x)
        loss = self.loss(activations[-1], a_hat)
        deltas = self.backward_pass(activations, a_hat)
        weight_derivatives = []

        for delta, acts in zip(deltas, [x] + activations):
            delta = delta.reshape((len(delta), 1))
            weight_derivatives.append(delta * acts)
        return weight_derivatives, deltas, loss

    def vanilla_gradient_descent(self, training_data:list, epochs:int, lr:float):
        """Gradient descent for training the neural network. 
        
        The weights and biases are updated only after going through the whole training data set.

        Args:
            training_data (list): List of tuples; tuples of np.ndarray; inputs and expected outputs.
            epochs (int): Number of times the data set is iterated through.
            lr (float): Learning rate.

        Returns:
            list: List of np.float64; mean loss value of each epoch.
        """
        learning_data = []
        n = len(training_data)
        training_data = training_data.copy()

        for _ in range(epochs):
            shuffle(training_data)

            gradient_wrt_weights = [np.zeros_like(w) for w in self.weights]
            gradient_wrt_biases = [np.zeros_like(b) for b in self.biases]
            loss_this_epoch = 0

            for inp, ex in training_data:
                weight_derivatives, bias_derivatives, loss = self.gradient_calculation(inp, ex)
                loss_this_epoch += loss

                for i in range(self.n_layers - 1):
                    gradient_wrt_weights[i] += weight_derivatives[i]
                    gradient_wrt_biases[i] += bias_derivatives[i]

            for i in range(self.n_layers - 1):
                gradient_wrt_weights[i] /= n
                gradient_wrt_biases[i] /= n
            loss_this_epoch /= n

            learning_data.append(loss_this_epoch)
            self.update_weights_and_biases(gradient_wrt_weights, gradient_wrt_biases, lr)

        return learning_data

    def stochastic_gradient_descent(self, training_data:list, epochs:int, lr:float):
        """Gradient descent for training the neural network.
        
        The weights and biases are updated after every training example.

        Args:
            training_data (list): List of tuples; tuples of np.ndarray; inputs and expected outputs.
            epochs (int): Number of times the data set is iterated through.
            lr (float): Learning rate.

        Returns:
            list: List of np.float64; mean loss value of each epoch.
        """
        learning_data = []
        n = len(training_data)
        training_data = training_data.copy()

        for _ in range(epochs):
            shuffle(training_data)

            loss_this_epoch = 0

            for inp, ex in training_data:
                weight_derivatives, bias_derivatives, loss = self.gradient_calculation(inp, ex)
                loss_this_epoch += loss
                self.update_weights_and_biases(weight_derivatives, bias_derivatives, lr)

            loss_this_epoch /= n
            learning_data.append(loss_this_epoch)

        return learning_data

    def minibatch_gradient_descent(self, training_data:list, minibatch_size:int, epochs:int, lr:float):
        """Gradient descent for training the neural network.
        
        The training data is split into batches and the weights and biases are updated after each one.

        Args:
            training_data (list): List of tuples; tuples of np.ndarray; inputs and expected outputs.
            minibatch_size (int): Number of training examples in one mini batch.
            epochs (int): Number of times the data set is iterated through.
            lr (float): Learning rate.

        Returns:
            list: List of np.float64; mean loss value of each epoch.
        """
        learning_data = []
        n = len(training_data)
        training_data = training_data.copy()

        for _ in range(epochs):
            shuffle(training_data)

            loss_this_epoch = 0
            minibatches = [training_data[i:i+minibatch_size] for i in range(0, n, minibatch_size)]

            for minibatch in minibatches:
                gradient_wrt_weights = [np.zeros_like(w) for w in self.weights]
                gradient_wrt_biases = [np.zeros_like(b) for b in self.biases]

                for inp, ex in minibatch:
                    weight_derivatives, bias_derivatives, loss = self.gradient_calculation(inp, ex)
                    loss_this_epoch += loss

                    for i in range(self.n_layers - 1):
                        gradient_wrt_weights[i] += weight_derivatives[i]
                        gradient_wrt_biases[i] += bias_derivatives[i]

                for i in range(self.n_layers - 1):
                    gradient_wrt_weights[i] /= minibatch_size
                    gradient_wrt_biases[i] /= minibatch_size

                self.update_weights_and_biases(gradient_wrt_weights, gradient_wrt_biases, lr)

            loss_this_epoch /= n
            learning_data.append(loss_this_epoch)

        return learning_data

    def update_weights_and_biases(self, new_w:list, new_b:list, lr:float):
        """Update all the weights and biases of the network to descend the gradient.

        Args:
            new_w (list): List of np.ndarray; weight derivatives.
            new_b (list): List of np.ndarray; bias derivatives.
            lr (float): Learning rate.
        """
        for i in range(self.n_layers - 1):
            self.weights[i] -= lr * new_w[i]
            self.biases[i] -= lr * new_b[i]

def glorot(n, m):
    """Weight initialization function suitable for neural network layers using the sigmoid activation function.

    Args:
        n (int): Number of inputs for this layer ("fan-in").
        m (int): Number of outputs (neurons) for this layer ("fan_out").

    Returns:
        np.ndarray: Weights for this layer.
    """
    b = sqrt(6 / (n + m))
    a = - b
    return np.random.uniform(a, b, (m, n))

def sigmoid(z:np.ndarray):
    """Activation function for the neural network to introduce nonlinearity.

    Args:
        z (np.ndarray): Weighted sum.

    Returns:
        np.ndarray: Array of values between zero and one.
    """
    return 1 / (1 + np.exp(-z))
