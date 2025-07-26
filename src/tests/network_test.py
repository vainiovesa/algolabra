import unittest
import numpy as np
from network import Network, save, load
from data_handling import get_test_data


class TestNetwork(unittest.TestCase):
    def setUp(self):
        self.layers = [10, 5, 6, 3, 4]
        self.net = Network(self.layers)
        n = self.layers[0]
        m = self.layers[-1]
        self.inputs1 = np.array([1 for _ in range(n)])
        self.inputs2 = np.array([0.5 for _ in range(n)])
        self.output1 = np.array([1 for _ in range(m)])
        self.output2 = np.array([0.2 for _ in range(m)])

        self.small_net = Network([2, 3, 1])
        self.data = [(np.array([1, 1]), np.array([0])),
                     (np.array([0, 1]), np.array([1])),
                     (np.array([1, 0]), np.array([1])),
                     (np.array([0, 0]), np.array([0]))]
        self.lr = 3

        self.test_data = get_test_data()
        self.mnist_net = Network([784, 10, 10])

    def test_weights_and_biases_right_type(self):
        for weight in self.net.weights:
            self.assertEqual(type(weight), np.ndarray)
        for bias in self.net.biases:
            self.assertEqual(type(bias), np.ndarray)

    def test_weights_and_biases_shaped_right(self):
        n = len(self.layers)
        for i in range(1, n):
            self.assertEqual(self.net.weights[i - 1].shape[0], self.layers[i])
            self.assertEqual(
                self.net.weights[i - 1].shape[1], self.layers[i - 1])
            self.assertEqual(self.net.biases[i - 1].shape[0], self.layers[i])

    def test_activations_right_type(self):
        activations = self.net.feed_forward(self.inputs1)
        for activation in activations:
            self.assertEqual(type(activation), np.ndarray)

    def test_activations_right_size(self):
        activations = [self.inputs1]
        activations += self.net.feed_forward(self.inputs1)
        for activation, layer in zip(activations, self.layers):
            self.assertEqual(len(activation), layer)

    def test_deltas_right_size(self):
        activations = self.net.feed_forward(self.inputs1)
        deltas = self.net._backward_pass(activations, self.output1)
        for d, a in zip(deltas, activations):
            self.assertEqual(d.shape, a.shape)

    def test_gradient_right_size(self):
        x, y = self.inputs1, self.output1
        weight_d, bias_d, _ = self.net._gradient_calculation(x, y)
        weights, biases = self.net.weights, self.net.biases
        for wd, bd, w, b in zip(weight_d, bias_d, weights, biases):
            self.assertEqual(wd.shape, w.shape)
            self.assertEqual(bd.shape, b.shape)

    def test_vanilla_gradient_descends(self):
        ep = 2000
        learning_data, _ = self.small_net.vanilla_gradient_descent(
            self.data, ep, self.lr)
        for i in range(1, ep):
            self.assertLess(learning_data[i], learning_data[i - 1])
        for x, y in self.data:
            rough_output = round(self.small_net.evaluate(x)[0])
            self.assertEqual(rough_output, y[0])

    def test_stochastic_gradient_descends(self):
        ep = 1000
        learning_data, _ = self.small_net.stochastic_gradient_descent(
            self.data, ep, self.lr)
        self.assertLess(learning_data[-1], learning_data[ep // 2])
        self.assertLess(learning_data[ep // 2], learning_data[0])
        for x, y in self.data:
            rough_output = round(self.small_net.evaluate(x)[0])
            self.assertEqual(rough_output, y[0])

    def test_minibatch_gradient_descends(self):
        ep = 1500
        mb_size = 2
        learning_data, _ = self.small_net.minibatch_gradient_descent(
            self.data, mb_size, ep, self.lr)
        self.assertLess(learning_data[-1], learning_data[ep // 2])
        self.assertLess(learning_data[ep // 2], learning_data[0])
        for x, y in self.data:
            rough_output = round(self.small_net.evaluate(x)[0])
            self.assertEqual(rough_output, y[0])

    def test_save_and_load_work(self):
        net1 = Network([2, 3, 2])
        save(net1, "test_neuralnetwork")
        net2 = load("test_neuralnetwork")
        for w1, w2 in zip(net1.weights, net2.weights):
            self.assertTrue(np.array_equal(w1, w2))
        for b1, b2 in zip(net1.biases, net2.biases):
            self.assertTrue(np.array_equal(b1, b2))

    def test_overall_loss_reasonable(self):
        data = [(self.inputs1, self.output1), (self.inputs2, self.output2)]
        loss = self.net.overall_loss(data)
        self.assertGreater(loss, 0)
        self.assertLess(loss, self.layers[-1] ** 2)

    def test_validation_accuracy_reasonable(self):
        acc = self.mnist_net.validation_accuracy(self.test_data)
        self.assertLessEqual(0, acc)
        self.assertLessEqual(acc, 1)

    def test_model_overfits_vanilla(self):
        for _ in range(100):
            _, accuracy_list = self.mnist_net.vanilla_gradient_descent(
                self.test_data, 3, 0.1, self.test_data)
            accuracy = accuracy_list[0]
            if accuracy == 1:
                break
        self.assertEqual(accuracy, 1)

    def test_model_overfits_stochastic(self):
        for _ in range(50):
            _, accuracy_list = self.mnist_net.stochastic_gradient_descent(
                self.test_data, 1, 0.1, self.test_data)
            accuracy = accuracy_list[0]
            if accuracy == 1:
                break
        self.assertEqual(accuracy, 1)

    def test_model_overfits_minibatch_and_classificationtest_works(self):
        corr, incorr = self.mnist_net.test_classification(self.test_data)
        self.assertEqual(len(corr) + len(incorr), len(self.test_data))
        for _ in range(50):
            _, accuracy_list = self.mnist_net.minibatch_gradient_descent(
                self.test_data, 10, 1, 1, self.test_data)
            accuracy = accuracy_list[0]
            if accuracy == 1:
                break
        self.assertEqual(accuracy, 1)
        corr, incorr = self.mnist_net.test_classification(self.test_data)
        self.assertEqual(len(corr), len(self.test_data))
        self.assertEqual(len(incorr), 0)
