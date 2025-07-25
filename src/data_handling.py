import gzip
import pickle
import numpy as np


def get_data(path: str = "data/mnist.pkl.gz"):
    """Get training, validation and testing data from the mnist data set.

    Format compatible with training a neural network of network.Network class.

    Args:
        path (str, optional): Path to mnist. Defaults to "data/mnist.pkl.gz".

    Returns:
        tuple: Tuple of three lists; list of tuple; tuple of np.ndarray; Inputs and expected outputs
        in format [(x, y), ...], ...
    """
    with gzip.open(path) as file:
        tr_data, va_data, te_data = pickle.load(file, encoding="latin1")

    training_data, validation_data, testing_data = data_converter(
        tr_data, va_data, te_data)
    return training_data, validation_data, testing_data


def data_converter(training_data, validation_data, testing_data):
    """Convert data into the format suitable for a neural network of network.Network class.
    """
    training_data_list, validation_data_list, testing_data_list = [], [], []

    for inputs, expected_ouput in zip(training_data[0], training_data[1]):
        expected_ouput = output_converter(expected_ouput)
        training_data_list.append((inputs, expected_ouput))

    for inputs, expected_ouput in zip(validation_data[0], validation_data[1]):
        expected_ouput = output_converter(expected_ouput)
        validation_data_list.append((inputs, expected_ouput))

    for inputs, expected_ouput in zip(testing_data[0], testing_data[1]):
        expected_ouput = output_converter(expected_ouput)
        testing_data_list.append((inputs, expected_ouput))

    return training_data_list, validation_data_list, testing_data_list


def output_converter(y: int):
    """Convert expected output from an integer to np.ndarray for classification.
    """
    return np.array([0 if i != y else 1 for i in range(9 + 1)])


def get_test_data(path: str = "data/mnist.pkl.gz"):
    """Get data for automated tests from the mnist data set.

    Args:
        path (str, optional): Path to mnist. Defaults to "data/mnist.pkl.gz".

    Returns:
        list: List of tuples; tuples of np.ndarray; Inputs and expected outputs in format
        [(x, y), ...]. Consists only of ones and twos.
    """
    with gzip.open(path) as file:
        _, data, _ = pickle.load(file, encoding="latin1")

    test_data = test_data_converter(data)
    return test_data


def test_data_converter(data):
    """Convert test data into the format suitable for a neural network of network.Network class.

    Returns a list of length 100 consisting only of ones and twos of the mnist data set.
    """
    test_data = []
    for x, y in zip(data[0], data[1]):
        if y in (1, 2):
            y = output_converter(y)
            test_data.append((x, y))
        if len(test_data) > 99:
            break
    return test_data
