import unittest
import numpy as np
from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.layers = [10, 5, 6, 3, 4]
        self.net = Network(self.layers)

    def test_weights_and_biases_right_type(self):
        for weight in self.net.weights:
            self.assertEqual(type(weight), np.ndarray)
        for bias in self.net.biases:
            self.assertEqual(type(bias), np.ndarray)

    def test_weights_and_biases_shaped_right(self):
        n = len(self.layers)
        for i in range(1, n):
            self.assertEqual(self.net.weights[i - 1].shape[0], self.layers[i])
            self.assertEqual(self.net.weights[i - 1].shape[1], self.layers[i - 1])
            self.assertEqual(self.net.biases[i - 1].shape[0], self.layers[i])

    def test_activations_right_type(self):
        inputs = np.array([1 for _ in range(self.layers[0])])
        activations = self.net.feed_forward(inputs)
        for activation in activations:
            self.assertEqual(type(activation), np.ndarray)

    def test_activations_right_size(self):
        inputs = np.array([1 for _ in range(self.layers[0])])
        activations = [inputs]
        activations += self.net.feed_forward(inputs)
        for activation, layer in zip(activations, self.layers):
            self.assertEqual(len(activation), layer)
