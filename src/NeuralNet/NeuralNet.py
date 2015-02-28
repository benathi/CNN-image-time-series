'''
Created on Feb 27, 2015

@author: Ben Athiwaratkun (pa338)

'''
from __future__ import division
import numpy as np
import os
import inspect
import collections
EPSILON = 0.12


class NeuralNet:
    ''' Required Fields '''
    Thetas = []
    trainData = None
    outputDim = 0
    inputDim = 0
    numInputs = 0
    actFns = []
    actGs = []
    numLayers = 0

    def sigmoid(self, z):
        return 1.0/(1+np.exp(-z))
    def sigmoidGrad(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    activationDicts = {}
    activationDicts['sigmoid'] = (sigmoid, sigmoidGrad)

    ''' Optional Fields '''
    testData = None
    
    
    def loadData(self, dataset='MNIST'):
        current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        if dataset == 'MNIST':
            designMatrix = np.loadtxt(os.path.join( current_folder_path,
                                                    '../../data/digits/digits_designMatrix.txt') )
            data_labels = np.loadtxt(os.path.join( current_folder_path,
                                                   '../../data/digits/digits_label.txt'))
        else:
            print "Other Data"
            
        ''' Adding column of 1's to X'''
        self.numInputs, self.inputDim = np.shape(designMatrix)
        designMatrix = np.c_[np.ones((self.numInputs,1)), designMatrix]
        Data = collections.namedtuple('Data', ['X','Y'])
        d = Data(designMatrix, data_labels)
        return d
            
    
    def __init__(self, trainingData='MNIST',
                 numClasses=10,
                 hiddenLayersSize=[25], 
                 activationFunctions=['sigmoid']):
        print 'Initializing Neural Network Object'
        self.outputDim = numClasses
        self.trainData = self.loadData(trainingData)
        print '1.Data Loaded.'
        print '2. Initializing Thetas.'
        self.numLayers = 1 + len(hiddenLayersSize)
        temp1 = ([self.inputDim] + hiddenLayersSize + [self.outputDim])
        for i in range(self.numLayers):
            hSize = temp1[i]
            hSizeNext = temp1[i+1]
            _Theta = np.random.uniform(-EPSILON,EPSILON,(hSizeNext, hSize+1))
            # randomize values
            self.Thetas.append(_Theta)
            #print np.shape(self.Thetas[i])
        ## activation function
        print '2. Mapping Activation Functions', activationFunctions
        for a in activationFunctions:
            if a in self.activationDicts:
                self.actFns, self.actGs = self.activationDicts[a]
            else:
                print 'Unsupported Activation Function:', a
                print 'Dictionary of Supported Functions:', self.activationDicts
    
    def gradient(self):
        pass
    
    def train(self, tolerance=0.01, regParams=[0.01]):
        print 'Training'
        numIt = 0
        alpha = 0.01    ## TODO - configurable
        costList = []
        while True:
            numIt += 1
            J,G = self.gradient(regParams)
            for i in range(len(self.Thetas)):
                self.Thetas[i] += alpha*G[i]
            costList.append(J)
            print 'Iteration %d: Cost = %f' % (numIt, J)
            if i > 30 and (costList[i-1]-J)/J < tolerance:
                break
    
    def hypothesis(self, X):
        h = np.array(X) # deep copy
        for i in range(self.numLayers):
            activation = self.actFns[i]
            if i > 0:   # assume X already has columns of 1
                h = np.c_(np.ones(()), h)
            h = activation(np.dot(h, self.Thetas[i].T ) )   # has to be numpy map function
        return h
    
    def classify(self, X):
        h = self.hypothesis(self, X)
        return  np.argmax(h,axis=1) # TODO - check if it's +1

def main():
    nn = NeuralNet()
    print [np.shape(ob) for ob in nn.Thetas]
    nn.train()
    print [np.shape(i) for i in nn.trainData]
    
    
    
if __name__ == "__main__":
    main()