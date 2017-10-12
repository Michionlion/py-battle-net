# Likely this object oriented implementation of a neural network will be replaced with a matrix forward propagation network at some point in the future, before actual implementation of the NN


class DNA:
    def __init__(self, encoded):
    
        self.supergenome = []
        self.genome = b''
    #
    
    def getweight(self, neuronId):
        return 0
    #
    
    def encode(self):
        encoded = "DNAOBJ_ENCODED:"
        
        return encoded
    #
#
class Neuron:
    def __init__(self, neuronId):
        self.id = neuronId
    #
#
class Layer:
    def __init__(self, neuronIds):
        self.neurons = []
        for id in neuronIds:
            self.neurons.append(Neuron(id))
    #
#
class NeuralNetwork:
    
    def __init__(self, dna):
        self.dna = read_DNA(dna)
        
        #construct based on DNA
        
        
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