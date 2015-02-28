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
    data_labels = None
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
        self.data_labels = data_labels
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
        return self.gradientLogLoss(X=self.trainData[0], y=self.trainData[1],
                                    regParams=args['regParams'])
    

    def gradientLogLoss(self, regParams, X, y):
        ## 1. Cost Function
        m = np.shape(X)[0]*1.0
        ## Initial w/o any hidden layers
        H = np.dot(X, self.Thetas[0].T)
        H = self.actFns[0](H)
        ## Now for each hidden layer:
        for i in range(1, len(self.actFns)):
            H = np.dot(np.concatenate((np.ones((m,1)), H), axis=1),
                       self.Thetas[i].T )
            H = self.actFns[i](H)
        ##  Unregularized Cost Function  
        J = np.sum(- np.multiply(y, np.log(H)) + np.multiply( y-1.0, np.log(1.0-H)) )/m
        #print "Unregularized cost", J
        for i in range(len(self.Thetas)):
            J += regParams[i]/(2.0*X.shape[0]) * np.sum(np.square(self.Thetas[i][:,1:]))
        #print "Regularized Cost", J
        
        ## 2. Gradient
        ## First, calculate initial values:
        ## zvec goes from z_2, z_3, ... 
        zvec = [np.dot(X,self.Thetas[0].T)]
        for i in range(1,len(self.Thetas)):
            zvec.append(  np.dot( np.column_stack( (np.ones((m,1)) ,self.actFns[i-1](zvec[i-1])) ),
                                         self.Thetas[i].T)   )
        delta_vec = [ self.actFns[-1](zvec[-1]) - y]
        for i in range(1, len(self.Thetas)):
            j = len(self.Thetas) - i - 1
            delta_vec.insert(0,
                             np.dot(delta_vec[0], self.Thetas[j+1])[:,1:] * self.actGs[j+1](zvec[j]))
        
        Theta_grads = [ (1.0/m) * np.dot(delta_vec[0].T, X )     ]
        for i in range(len(self.Thetas)-1):
            Theta_grads.append( (1.0/m)*np.dot( delta_vec[i+1].T,
                                                np.column_stack( (np.ones(np.shape(zvec[i])[0]), self.actFns[i](zvec[i]))) )
                                                )
        for i in range(len(self.Thetas)-1):
            Theta_grads[i] += (regParams[i]/m)*np.column_stack( ((np.zeros(np.shape(self.Thetas[i])[0])),
                                                                self.Thetas[i][:,1:]))
            #Theta_grads[j] += (regParams[i]/m)*self.Thetas[i]
        
        return (J,Theta_grads)
        
    
    def train(self, tolerance=0.0001, maxNumIts = 10000, regParams=[1.0,1.0]):
        print 'Training'
        numIt = -1
        alpha = 0.3    ## TODO - configurable
        costList = []
        while numIt < maxNumIts:
            numIt += 1
            J,G = self.gradient(regParams=regParams)
            for i in range(len(self.Thetas)):
                self.Thetas[i] -= alpha*G[i]
            costList.append(J)
            print 'Iteration %d: Cost = %f' % (numIt, J)
            if numIt > 30:
                percentTage = (costList[numIt-1]-costList[numIt])/J*100.0
                #print 'Decreased by %f percent' % percentTage
                if percentTage < tolerance:
                    break
    
    def hypothesis(self, X):
        h = np.array(X) # deep copy
        for i in range(self.numLayers):
            activation = self.actFns[i]
            if i > 0:   # assume X already has columns of 1
                h = np.column_stack((np.ones((np.shape(X)[0],1)), h))
            h = activation(np.dot(h, self.Thetas[i].T ) )   # has to be numpy map function
        return h
    
    def classify(self, X):
        h = self.hypothesis(X)
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
        
        nRow1, nCol1 = np.shape(Theta1)
        nRow2, nCol2 = np.shape(Theta2)
        DEL = 0.000001
        diff_error = 0.0
        for r in range(nRow1):
            for c in range(nCol1):
                print '(r,c)=(%d,%d)' % (r,c)
                J_before, G_before = self.gradient(X=self.trainData[0], y=self.trainData[1])
                self.Thetas[0][r,c] += DEL
                J_after, G_after = self.gradient(X=self.trainData[0], y=self.trainData[1])
                diff_error += ( (J_after-J_before)/DEL - G_before[0][r,c] )**2
                print 'Accumulated Error = ', diff_error
        
        

def main():
    nn = NeuralNet(hiddenLayersSize=[25, 15], 
                 activationFunctions=['sigmoid', 'sigmoid', 'sigmoid'])
    print [np.shape(ob) for ob in nn.Thetas]
    nn.train(maxNumIts=10000,regParams=[0.1,0.1,0.1])
    print [np.shape(i) for i in nn.trainData]
    #nn.test_loadSampleThetas()
    #print(nn.trainData[1] )
    #print(nn.classify(nn.trainData[0]))
    print np.sum((nn.data_labels != nn.classify(nn.trainData[0])))/nn.numInputs
    
    
if __name__ == "__main__":
    main()