from tensorflow.keras.datasets.mnist import load_data
import build_network
import numpy as np
import matplotlib.pyplot as plt
from time import time


def plot_cost_func(costs):
    """
    Plot the cost over training steps graph
    :param costs: list of all costs
    """
    plt.plot([i * 100 for i in range(len(costs))], costs)
    plt.xlabel("Training step")
    plt.ylabel("Cost")
    plt.title("Validation costs")
    plt.legend()
    plt.show()


def normalise_data(x, y):
    """
    performs reshape & normalization to the mnist data
    :param x: mnist digits data
    :param y: mnist digits labels
    :return:
    x_normalized, y_normalized - prepared data to run
    """
    num_of_classes = 10
    pixel_value = 255.0
    x_normalized = np.array([(x / pixel_value).flatten() for x in x])
    y = y.flatten().astype(int)
    y_normalized = np.zeros((y.shape[0], num_of_classes))
    y_normalized[np.arange(y.shape[0]), y] = 1

    return x_normalized, y_normalized


def prep_data():
    """
    loads and splits the mnist data into train & test subsets,
    then prepares the data for the neural network
    :return:
    x_train, y_train, x_test, y_test - data to run
    """
    (x_train, y_train), (x_test, y_test) = load_data(path='mnist.npz')
    x_train, y_train = normalise_data(x_train, y_train)
    x_test, y_test = normalise_data(x_test, y_test)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    np.random.seed(33)
    x_train, y_train, x_test, y_test = prep_data()
    t_start = time()
    parameters, costs = build_network.L_layer_model(x_train.T, y_train.T, layers_dims=[784, 20, 7, 5, 10],
                                                    learning_rate=0.009,
                                                    num_iterations=10, batch_size=64)
    t_end = time()
    print(f"Final accuracy score on Test data: {build_network.predict(x_test.T, y_test.T, parameters)}")
    plot_cost_func(costs)

    print("Running time: ", t_end - t_start)
