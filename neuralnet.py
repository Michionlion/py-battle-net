# Likely this object oriented implementation of a neural network will be replaced with a matrix forward propagation network at some point in the future, before actual implementation of the NN

import numpy as np

class NeuralNetwork:
    
    def __init__(self, layerSize):
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize
        self.weights = []
        
        self._layerInput = []
        self._layerOutput = []
        
        #create weight matrixes
        for (inp, out) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=1, size=(out, inp+1)))
            
        #
    #
    
    def propagate(self, input):
        # assume input is in form [[x1, y1, z1],[x2,y2,z2]] where each row should be transposed into column matrix/vector for propagation, and multiple rows in the input map to different vectors for parallel input running.
        
        numCases = input.shape[0]
        
        #clear out previous layer in/out lists
        self._layerInput = []
        self._layerOutput = []
        
        for index in range(self.layerCount):
            if index == 0:
                #transpose to columns, add 1 to bottom, so matrix will look like below with numCases columns
                # x1 x2 ...
                # y1 y2 ...
                # z1 z2 ...
                #  1  1 ...
                matIn = np.vstack([input.T, np.ones([1, numCases])])
                layerInput = self.weights[0].dot(matIn)
            else:
                #take last layer output and do the same thing as done with matFirstInput
                matIn = np.vstack([self._layerOutput[-1], np.ones([1, numCases])])
                layerInput = self.weights[index].dot(matIn)
            
            #
            #print("before relu: " + str(layerInput))
            #print("relu: " + str(relu(layerInput)))
            #print("after relu: " + str(layerInput))
            
            self._layerInput.append(layerInput)
            #self._layerOutput.append(relu(layerInput))
            
            #do softmax for last step
            if index < self.layerCount-1:
                self._layerOutput.append(relu(layerInput))
            else:
                #act = np.frompyfunc(softmax, 1, 1);
                self._layerOutput.append(sigmoid(layerInput))
        #
        
        return self._layerOutput[-1].T
        
    #
    
#

#activation functions, returns derivative too if needed with option deriv
def relu(x):
    return np.maximum(x, 0)
    #return x if x > 0 else 0
# relu
def softmax(w, t=1.0):
    e = np.exp(np.array(x) / t)
    return e / np.sum(e)
#
def sigmoid(x):
    return 1/(1+np.exp(-x))
# sigmoid


# if run as script, exec test
if __name__ == "__main__":
    nn = NeuralNetwork((5, 8, 8, 5))
    print(nn.shape)
    print(nn.weights)
    
    input = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]])
    output = nn.propagate(input)
    
    print("\nInputL {0}\nOutput: {1}".format(input, output))