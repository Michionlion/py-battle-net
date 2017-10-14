# Likely this object oriented implementation of a neural network will be replaced with a matrix forward propagation network at some point in the future, before actual implementation of the NN

import numpy as np
#
# Activation Functions
#
def relu(x):
    return np.maximum(x, 0)
    #return x if x > 0 else 0
#
def softmax(x, t=1.0):
    e = np.exp(x / t)
    return e / np.sum(e)
#
def sigmoid(x):
    return 1/(1+np.exp(-x))
#

#
# Gene to weight matrix list conversion
#




#
# Feed-Forward Neural Network
#
class NeuralNetwork:
    
    def __init__(self, shape, weights=None):
        self.layerCount = len(shape) - 1
        self.shape = shape
        
        self._layerInput = []
        self._layerOutput = []
        
        if weights == None:
            self.weights = []
            #create weight matrixes
            for (inp, out) in zip(shape[:-1], shape[1:]):
                self.weights.append(np.random.normal(scale=0.2, size=(out, inp+1)))
            #
        else:
            self.weights = weights
    #
    
    def evaluate(self, input):
        # assume input is in form [x, y, z, ...] or np.array([x, y, z, ...]) or [[x, y, z, ...]]
        # need to make sure it is 2d, otherwise transpose will not work (transpose of 1d is itself, not column vec)
        input = np.array(input, ndmin=2)
        #clear out previous layer in/out lists
        self._layerInput = []
        self._layerOutput = []
        
        for index in range(self.layerCount):
            if index == 0:
                #transpose to columns, add 1 to bottom, so vector will
                #look like: x y z ... 1 vertically
                vecIn = np.vstack((input.T, [1]))
                layerInput = self.weights[0].dot(vecIn)
            else:
                #take last layer output and do the same thing as done with before
                vecIn = np.vstack((self._layerOutput[-1], [1]))
                #mul weights with inputs for this layer
                layerInput = self.weights[index].dot(vecIn)
            
            #
            
            self._layerInput.append(layerInput)
            
            #do softmax for last step
            if index < self.layerCount-1:
                self._layerOutput.append(relu(layerInput))
            else:
                self._layerOutput.append(softmax(layerInput.T).T)
        #
        
        #return as a numpy array, but only the output column formatted as a row
        return self._layerOutput[-1].T[0]
        
    #
    
#

# return network info
def netinfo(network):
    info = "NETWORK SHAPE: " + str(nn.shape) + ", WEIGHTS:\n"
    for i in range(len(nn.weights)):
        mat = nn.weights[i]
        info += "Weight matrix with shape {0!s} for transfer from layer {1!s} to layer {2!s}:\n".format(mat.shape, i, i+1)
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
    nn = NeuralNetwork((5, 8, 8, 5))
    
    print(netinfo(nn))
    
    input = [0, 1, -1, -1, -1]
    output = nn.evaluate(input).tolist()
    
    info = "Input: "
    for s in input: info += " {0:5.1f}".format(s)
    info += "\nOutput:"
    for s in output: info += " {0:5.3f}".format(s)
    
    print(info)