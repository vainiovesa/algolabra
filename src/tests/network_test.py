import unittest
import numpy as np
from network import Network

class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.layers = [10, 5, 6, 3, 4]
        self.net = Network(self.layers)
        n = self.layers[0]
        m = self.layers[-1]
        self.inputs = np.array([1 for _ in range(n)])
        self.output = np.array([1 for _ in range(m)])

        self.small_net = Network([2, 3, 1])
        self.data = [(np.array([1, 1]), np.array([0])),
                     (np.array([0, 1]), np.array([1])),
                     (np.array([1, 0]), np.array([1])),
                     (np.array([0, 0]), np.array([0]))]
        self.lr = 3

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
        activations = self.net.feed_forward(self.inputs)
        for activation in activations:
            self.assertEqual(type(activation), np.ndarray)

    def test_activations_right_size(self):
        activations = [self.inputs]
        activations += self.net.feed_forward(self.inputs)
        for activation, layer in zip(activations, self.layers):
            self.assertEqual(len(activation), layer)

    def test_deltas_right_size(self):
        activations = self.net.feed_forward(self.inputs)
        deltas = self.net.backward_pass(activations, self.output)
        for d, a in zip(deltas, activations):
            self.assertEqual(d.shape, a.shape)

    def test_gradient_right_size(self):
        x, y = self.inputs, self.output
        weight_d, bias_d, _ = self.net.gradient_calculation(x, y)
        weights, biases = self.net.weights, self.net.biases
        for wd, bd, w, b in zip(weight_d, bias_d, weights, biases):
            self.assertEqual(wd.shape, w.shape)
            self.assertEqual(bd.shape, b.shape)

    def test_vanilla_gradient_descends(self):
        ep = 2000
        learning_data = self.small_net.vanilla_gradient_descent(self.data, ep, self.lr)
        for i in range(1, ep):
            self.assertLess(learning_data[i], learning_data[i - 1])
        for x, y in self.data:
            rough_output = round(self.small_net.evaluate(x)[0])
            self.assertEqual(rough_output, y[0])

    def test_stochastic_gradient_descends(self):
        ep = 1000
        learning_data = self.small_net.stochastic_gradient_descent(self.data, ep, self.lr)
        self.assertLess(learning_data[-1], learning_data[ep // 2])
        self.assertLess(learning_data[ep // 2], learning_data[0])
        for x, y in self.data:
            rough_output = round(self.small_net.evaluate(x)[0])
            self.assertEqual(rough_output, y[0])

    def test_minibatch_gradient_descends(self):
        ep = 1500
        mb_size = 2
        learning_data = self.small_net.minibatch_gradient_descent(self.data, mb_size, ep, self.lr)
        self.assertLess(learning_data[-1], learning_data[ep // 2])
        self.assertLess(learning_data[ep // 2], learning_data[0])
        for x, y in self.data:
            rough_output = round(self.small_net.evaluate(x)[0])
            self.assertEqual(rough_output, y[0])
