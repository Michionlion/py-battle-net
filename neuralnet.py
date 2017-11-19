"""Feed-Forward Neural Network, Activation Functions, and test script."""

import numpy as np
import pickle


def relu(x):
    """Relu Activation Function, returns x when x >= 0."""
    return np.maximum(x, 0)
#

import numpy as np
def softmax(x):
    """Soft-max Function, normalizes a numpy array so that the entries sum to 1."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
#


def sigmoid(x):
    """Sigmoid Activation Function, returns 0-1f."""
    return 1 / (1 + np.exp(-x))
#


def unflatten(shape, list):
    """Unflatten a list into a shaped numpy array."""
    index = 0
    weights = []
    for (inp, out) in zip(shape[:-1], shape[1:]):
        # reconstruct the matrix
        matrix = np.empty((out, inp + 1), dtype=float)
        for row in range(out):
            for entry in range(inp + 1):
                matrix[row][entry] = list[index]
                index += 1
        weights.append(matrix)
    return weights
#


def flatten(matrices):
    """Flatten a numpy array into a list."""
    list = []
    for matrix in matrices:
        for row in matrix:
            for weight in row:
                list.append(weight)
    return list
#


class NeuralNetwork:
    """Feed-Forward Neural Network."""

    def __init__(self, shape, weights=None):
        """Initialize the Neural Network."""
        self.layerCount = len(shape) - 1
        self.shape = shape

        self._layerInput = []
        self._layerOutput = []

        if weights is None:
            self.weights = []
            # create weight matrixes
            for (inp, out) in zip(shape[:-1], shape[1:]):
                self.weights.append(
                    np.random.normal(scale=1, size=(out, inp + 1)))
            #
        else:
            self.weights = weights

    def evaluate(self, input):
        """
        Execute forward propagation.

        We assume input is in form [x, y, z, ...], or np.array([x, y, z, ...]), or [[x, y, z, ...]]. We need to make sure it is 2d, otherwise transpose will not work (transpose of 1d is itself, not column vec).
        """
        input = np.array(input, ndmin=2)
        # clear out previous layer in/out lists
        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                # transpose to columns, add 1 to bottom, so vector will
                # look like: x y z ... 1 vertically
                vecIn = np.vstack((input.T, [1]))
                layerInput = self.weights[0].dot(vecIn)
            else:
                # take last layer output and do the same thing
                vecIn = np.vstack((self._layerOutput[-1], [1]))
                # mul weights with inputs for this layer
                layerInput = self.weights[index].dot(vecIn)

            #

            self._layerInput.append(layerInput)

            # do soft-max for last step
            if index < self.layerCount - 1:
                self._layerOutput.append(relu(layerInput))
            else:
                self._layerOutput.append(softmax(layerInput.T).T)
        #

        # return as a numpy array, but formatted as a row instead of column
        return self._layerOutput[-1].T[0]


def netinfo(network):
    """TODO: document this."""
    info = "NETWORK SHAPE: " + str(network.shape) + ", WEIGHTS:\n"
    for i in range(len(network.weights)):
        mat = network.weights[i]

        info += "Weight matrix with shape " + str(mat.shape)
        info += " for transfer from layer " + str(i)
        info += " to layer " + str(i + 1) + ":\n"

        for row in mat:
            info += "\n"
            for w in row:
                info += "{0:10.5f}".format(w)
            info += "\n"
        info += "\n- - - -\n\n"
    #
    return info
#


# if run as script, exec test
if __name__ == "__main__":
    nn = NeuralNetwork((2, 4, 2))

    print(netinfo(nn))

    input = [0, 1]
    output = nn.evaluate(input).tolist()

    info = "Input: "
    for s in input:
        info += " {0:5.1f}".format(s)
    info += "\nOutput:"
    for s in output:
        info += " {0:5.3f}".format(s)

    print(info)

    with open("weights.dat", 'wb') as file:
        pickle.dump(nn.weights, file, protocol=-1)
