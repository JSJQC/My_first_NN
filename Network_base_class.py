import numpy
import math

## Neural Network Zone ----------------------------------------------------------------------------

def activationFunction(layer):

        layerList = []
        for x in range(0, len(layer)):
                ## print(x)
                fx = 1 / (1 + math.exp(-layer[x]))
                layerList.append(fx)
            
        layerFinal = numpy.array(layerList)
        return layerFinal

class Neural_Network:
    
        def __init__(self):

                ## Each layer will be a numpy matrix
                
                self.fitness = None
                self.birthGeneration = None
                self.inputToHidden = numpy.random.uniform((-3), (3), (18,9))
                self.hiddenToOutput = numpy.random.uniform((-3), (3), (9, 18))
        
            
        def feedForward(self, inputs): ## Inputs will be the current board state saved as a list

                inputLayer = numpy.array(inputs)
                ## print (inputLayer)
                ## print ("")

                hiddenLayerRaw = numpy.matmul(self.inputToHidden, inputLayer)
                ## print (hiddenLayerRaw)
                ## print ("")

                hiddenLayer = activationFunction(hiddenLayerRaw)
                ## print (hiddenLayer)
                ## print ("")

                outputLayerRaw = numpy.matmul(self.hiddenToOutput, hiddenLayer)
                ## print (outputLayerRaw)
                ## print ("")

                outputLayer = activationFunction(outputLayerRaw)
                ## print (outputLayer)

                return outputLayer

## ------------------------------------------------------------------------------------------------
