from hypothesis import strategies as st, given, settings
import numpy as np
from data_handling import get_test_data
from network import Network


DATA = get_test_data()

@st.composite
def make_net(draw):
    layers = [784]
    layers += draw(st.lists(st.integers(1, 784), min_size=0, max_size=10))
    layers += [10]
    net = Network(layers)
    return net

@given(make_net())
@settings(deadline=None)
def test_nets_params_change_vanilla(net):
    weights1 = []
    for w in net.weights:
        weights1.append(w.copy())
    biases1 = []
    for b in net.biases:
        biases1.append(b.copy())

    net.vanilla_gradient_descent(DATA, epochs=1, lr=3)

    weights2 = []
    for w in net.weights:
        weights2.append(w.copy())
    biases2 = []
    for b in net.biases:
        biases2.append(b.copy())

    for w1, w2 in zip(weights1, weights2):
        assert not np.array_equal(w1, w2)
    for b1, b2 in zip(biases1, biases2):
        assert not np.array_equal(b1, b2)

@given(make_net())
@settings(deadline=None)
def test_nets_params_change_stochastic(net):
    weights1 = []
    for w in net.weights:
        weights1.append(w.copy())
    biases1 = []
    for b in net.biases:
        biases1.append(b.copy())

    net.stochastic_gradient_descent(DATA, epochs=1, lr=3)

    weights2 = []
    for w in net.weights:
        weights2.append(w.copy())
    biases2 = []
    for b in net.biases:
        biases2.append(b.copy())

    for w1, w2 in zip(weights1, weights2):
        assert not np.array_equal(w1, w2)
    for b1, b2 in zip(biases1, biases2):
        assert not np.array_equal(b1, b2)

@given(make_net())
@settings(deadline=None)
def test_nets_params_change_minibatch(net):
    weights1 = []
    for w in net.weights:
        weights1.append(w.copy())
    biases1 = []
    for b in net.biases:
        biases1.append(b.copy())

    net.minibatch_gradient_descent(DATA, minibatch_size=10, epochs=1, lr=3)

    weights2 = []
    for w in net.weights:
        weights2.append(w.copy())
    biases2 = []
    for b in net.biases:
        biases2.append(b.copy())

    for w1, w2 in zip(weights1, weights2):
        assert not np.array_equal(w1, w2)
    for b1, b2 in zip(biases1, biases2):
        assert not np.array_equal(b1, b2)
