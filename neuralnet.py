# Likely this object oriented implementation of a neural network will be replaced with a matrix forward propagation network at some point in the future, before actual implementation of the NN

import numpy as np

class NeuralNetwork:
    
    def __init__(self, layerSize):
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize
        
        self._layerInput = []
        self._layerOutput = []
        
        
        
    #
#



def read_DNA(dna):
    if dna.startswith("DNAOBJ_ENCODED:"):
        #actual dna string
        return DNA(dna)
    else:
        #its a file name
        with open(dna, 'r') as file:
            return DNA(file.read())
    #
#