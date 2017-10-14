# Likely this object oriented implementation of a neural network will be replaced with a matrix forward propagation network at some point in the future, before actual implementation of the NN

import numpy as np

class NeuralNetwork:
    
    def __init__(self, layerSize, weights=None):
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize
        
        self._layerInput = []
        self._layerOutput = []
        
        if weights == None:
            self.weights = []
            #create weight matrixes
            for (inp, out) in zip(layerSize[:-1], layerSize[1:]):
                self.weights.append(np.random.normal(scale=0.2, size=(out, inp+1)))
            #
        else:
            self.weights = weights
    #
    
    def evaluate(self, input):
        # assume input is in form [x, y, z, ...] or np.array([x, y, z, ...]) or [[x, y, z, ...]]
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

#activation functions, returns derivative too if needed with option deriv
def relu(x):
    return np.maximum(x, 0)
    #return x if x > 0 else 0
# relu
def softmax(x, t=1.0):
    e = np.exp(x / t)
    return e / np.sum(e)
#
def sigmoid(x):
    return 1/(1+np.exp(-x))
# sigmoid


# if run as script, exec test
if __name__ == "__main__":
    nn = NeuralNetwork((3, 3, 2))
    print("SHAPE: " + str(nn.shape))
    wshape = []
    for arr in nn.weights:
        wshape.append(arr.shape)
    #
    print("WEIGHTS SHAPE: " + str(wshape) + "\nWEIGHTS: " + str(nn.weights))
    
    input = [0, 1, 2]
    output = nn.evaluate(input).tolist()
    
    print("\nInput: " + str(input) + "\nOutput: " + str(output))