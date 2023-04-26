"""
Performs backward propagation
"""
import numpy as np


def linear_backward(dZ, cache):
    """
    Implements the linear part of the backward propagation process for a single layer
    :param dZ: the gradient of the cost with respect to the linear output of the current layer (layer l)
    :param cache:tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    m = cache['A'].shape[0]

    dA_prev = np.matmul(cache['W'].transpose(), dZ)
    # dA_prev = np.matmul(dZ.T, cache['W']).T / m
    dW_curr = np.matmul(dZ, cache['A'].transpose()) / m
    db_curr = np.sum(dZ, axis=1) / m
    db_curr = np.expand_dims(db_curr, axis=1)

    return dA_prev, dW_curr, db_curr


def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function first computes dZ and then applies
    the linear_backward function.
    Some comments:
        * The derivative of ReLU is f^' (x)={■(1&if x>0@0&otherwise)┤
        * The derivative of the softmax function is: p_i-y_i, where p_i is the softmax-adjusted probability of the class
        and y_i is the “ground truth” (i.e. 1 for the real class, 0 for all others)
        * You should use the activations cache created earlier for the calculation of the activation derivative and the linear
        cache should be fed to the linear_backward function
    :param dA: post activation gradient of the current layer
    :param cache: contains both the linear cache and the activations cache
    :param activation:
    :return:
        dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW – Gradient of the cost with respect to W (current layer l), same shape as W
        db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    if activation == 'relu':
        activation_function = relu_backward
    else:
        activation_function = softmax_backward

    dz = activation_function(dA, cache)
    dA_prev, dW, db = linear_backward(dz, cache)

    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    """
    Implements backward propagation for a ReLU unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return:
        dZ – gradient of the cost with respect to Z
    """
    Z = activation_cache['Z']
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0

    return dZ


def softmax_backward(dA, activation_cache):
    """
    Implements backward propagation for a softmax unit
    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)-  as 'AL'
    :return:
        dZ – gradient of the cost with respect to Z
    """
    # we are using the activation cache we updated before using softmax backward,
    # to have the softmax probabilities (AL), and the true labels (Y)
    softmax_probs = activation_cache['AL']
    y_labels = activation_cache['Y']
    dZ = softmax_probs - y_labels

    return dZ


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation process for the entire network.
    Some comments:
    the backpropagation for the softmax function should be done only once as only the output layers uses it and the RELU
    should be done iteratively over all the remaining layers of the network.
    :param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y: the true labels vector (the "ground truth" - true classifications)
    :param caches: list of caches containing for each layer: a) the linear cache; b) the activation cache
    :return:
        Grads - a dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    dA_prev = None
    num_of_layers = len(caches)

    # store the softmax probabilities (AL), and the true labels (Y)
    # in the activation cache (last layer) before applying softmax_backward
    caches[num_of_layers - 1]['AL'] = AL
    caches[num_of_layers - 1]['Y'] = Y

    for layer in range(num_of_layers, 0, -1):
        activation = 'relu' if dA_prev is not None else 'softmax'
        dA = dA_prev if dA_prev is not None else -(Y / AL) + (1 - Y) / (1 - AL)
        dA_prev, dW, db = linear_activation_backward(dA, caches[layer - 1], activation)
        grads.update({f'dA{layer}': dA, f'dW{layer}': dW, f'db{layer}': db})

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :param grads: a python dictionary containing the gradients (generated by L_model_backward)
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :return:
        parameters – the updated values of the parameters object provided as input
    """
    num_of_layers = len(parameters) // 2
    for layer in range(1, num_of_layers + 1):
        parameters[f'b{layer}'] -= learning_rate * grads[f'db{layer}'][0]
        parameters[f'W{layer}'] -= learning_rate * grads[f'dW{layer}']

    return parameters
