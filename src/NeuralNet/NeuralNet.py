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

def sigmoid(z):
    return 1.0/(1+np.exp(-z))
def sigmoidGrad(z):
    return sigmoid(z)*(1-sigmoid(z))

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
            data_labels = np.vectorize(lambda (x): (x -1))(data_labels)
        else:
            print "Other Data"
            
        ''' Adding column of 1's to X'''
        self.numInputs, self.inputDim = np.shape(designMatrix)
        designMatrix = np.column_stack( (np.ones((self.numInputs)), designMatrix) )
        label_matrix = np.zeros((self.numInputs, self.outputDim))
        for i in range(self.numInputs):
            label_matrix[i,data_labels[i]] = 1
        Data = collections.namedtuple('Data', ['X','Y'])
        d = Data(designMatrix, label_matrix)
        return d
            
    
    def __init__(self, trainingData='MNIST',
                 numClasses=10,
                 hiddenLayersSize=[25], 
                 activationFunctions=['sigmoid', 'sigmoid']):
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
        ## activation function
        print '2. Mapping Activation Functions', activationFunctions
        for a in activationFunctions:
            if a in self.activationDicts:
                _func, _grad = self.activationDicts[a]
                self.actFns.append(_func)
                self.actGs.append(_grad)
            else:
                print 'Unsupported Activation Function:', a
                print 'Dictionary of Supported Functions:', self.activationDicts
    
    def gradient(self, **args):
        print type(self.trainData[0])
        return self.gradientLogLoss(X=self.trainData[0], y=self.trainData[1], regParams=[1.0,1.0])
    

    def gradientLogLoss(self, X, y, regParams):
        ## Initial w/o any hidden layers
        #H = np.mat(X) * np.mat(self.Thetas[0].T)
        H = np.dot(X, self.Thetas[0].T)
        H = self.actFns[0](H)
        ## Now for each hidden layer:
        for i in range(1, len(self.actFns)):
            print 'i=', i
            H = np.dot(np.concatenate((np.ones((self.numInputs,1)), H), axis=1),
                       self.Thetas[i].T )
            H = self.actFns[i](H)
        ##  Unregularized Cost Function  
        J = np.sum(- np.multiply(y, np.log(H)) + np.multiply( y-1.0, np.log(1.0-H)) )/self.numInputs

        print "Unregularized cost", J

        ## Regularized Cost Function
        for i in range(len(self.Thetas)):
            J += regParams[i]/(2.0*X.shape[0]) * np.sum(np.square(self.Thetas[i][:,1:]))
        
        print "Regularized Cost", J
        
        print H[-1,:]
        return [J,[1]]
        
    
    def train(self, tolerance=0.01, maxNumIts = 1000, regParams=[0.01]):
        print 'Training'
        numIt = 0
        alpha = 0.01    ## TODO - configurable
        costList = []
        while True:
            numIt += 1
            J,G = self.gradient(regParams=regParams)
            for i in range(len(self.Thetas)):
                self.Thetas[i] += alpha*G[i]
            costList.append(J)
            print 'Iteration %d: Cost = %f' % (numIt, J)
            if (numIt > 30 and (costList[numIt-1]-J)/J < tolerance) or numIt > 1000:
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


    def test_loadSampleThetas(self):
        import os
        import inspect
        current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        Theta1 = np.loadtxt(os.path.join( current_folder_path,
                                          '../../data/digits/digits_Theta1.txt'), delimiter=',')
        Theta2 = np.loadtxt(os.path.join( current_folder_path, 
                                          '../../data/digits/digits_Theta2.txt'), delimiter=',')
        self.Thetas = [Theta1, Theta2]
        
        print self.gradient()

def main():
    nn = NeuralNet()
    print [np.shape(ob) for ob in nn.Thetas]
    #nn.train()
    print [np.shape(i) for i in nn.trainData]
    nn.test_loadSampleThetas()
    
    
    
if __name__ == "__main__":
    main()