'''
Created on Feb 27, 2015

@author: Ben Athiwaratkun (pa338)

'''
from __future__ import division
import numpy as np
import os
import inspect
import collections
from fmincg import fmincg
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
    data_labels_dict = None # dictionary mapping index to data attribute
    labelMapText = None     # dictionary mapping index to text label

    activationDicts = {}
    activationDicts['sigmoid'] = (sigmoid, sigmoidGrad)

    ''' Optional Fields '''
    testData = None
    
    
    def loadData(self, dataset='DIGITS'):
        import pickle
        current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        if dataset == 'DIGITS':
            designMatrix = np.loadtxt(os.path.join( current_folder_path,
                                                    '../../data/digits/digits_designMatrix.txt') )
            data_labels = np.loadtxt(os.path.join( current_folder_path,
                                                   '../../data/digits/digits_label.txt'))
            data_labels = np.vectorize(lambda (x): (x -1))(data_labels)
        if dataset == 'PLANKTON':
            print 'Loading Plankton Data'
            designMatrix , data_labels, self.data_labels_dict, self.labelMapText =  \
            pickle.load( open(os.path.join(current_folder_path,
                                           '../../data/planktonTrain.p'), 'rb'))
            designMatrix = 255 - designMatrix
            print 'Done Loading Plankton Data'
        else:
            print "Other Data"
            
        ''' Adding column of 1's to X'''
        self.data_labels = data_labels
        self.outputDim = int(np.max(data_labels)) + 1   # assume index starts from 0
        self.numInputs, self.inputDim = np.shape(designMatrix)
        designMatrix = np.column_stack( (np.ones((self.numInputs)), designMatrix) )
        label_matrix = np.zeros((self.numInputs, self.outputDim))
        for i in range(self.numInputs):
            label_matrix[i,data_labels[i]] = 1
        Data = collections.namedtuple('Data', ['X','Y'])
        d = Data(designMatrix, label_matrix)
        return d
            
    
    def __init__(self, trainingData='PLANKTON',
                 numClasses=10,
                 hiddenLayersSize=[25], 
                 activationFunctions=['sigmoid', 'sigmoid']):
        print 'Initializing Neural Network Object'
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
        self.Thetas = np.array(self.Thetas)
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
    
    def regCost(self, Thetas, X, y, regParams):
        return self.regCost_LogLoss(Thetas, X, y, regParams);
    def regGradient(self, Thetas, X, y, regParams):
        return self.regGradient_LogLoss(Thetas, X, y, regParams)
    
    def regGradientWithCost(self, Ts, X, y, regParams):
        #return self.regGradientWithCost_LogLoss(X=self.trainData[0], y=self.trainData[1],
        #                            regParams=args['regParams'])
        return self.regGradientWithCost_LogLoss(Ts, X, y, regParams)
    
    def regCost_LogLoss(self, Ts, X, y, regParams):
        ## 1. Cost Function
        m = np.shape(X)[0]*1.0
        H = self.hypothesis(X)
        ##  Unregularized Cost Function  
        J = self.LogLossScore_UnNormalized(H, y)
        #print "Unregularized cost", J
        for i in range(len(Ts)):
            J += regParams[i]/(2.0*X.shape[0]) * np.sum(np.square(Ts[i][:,1:]))
        return J
    
    def regGradient_LogLoss(self, Ts, X, y, regParams):
        m = np.shape(X)[0]*1.0
        ## 2. Gradient
        ## First, calculate initial values:
        ## zvec goes from z_2, z_3, ... 
        zvec = [np.dot(X,Ts[0].T)]
        for i in range(1,len(Ts)):
            zvec.append(  np.dot( np.column_stack( (np.ones((m,1)) ,self.actFns[i-1](zvec[i-1])) ),
                                         Ts[i].T)   )
        delta_vec = [ self.actFns[-1](zvec[-1]) - y]
        for i in range(1, len(Ts)):
            j = len(Ts) - i - 1
            delta_vec.insert(0,
                             np.dot(delta_vec[0], Ts[j+1])[:,1:] * self.actGs[j+1](zvec[j]))
        
        Theta_grads = [ (1.0/m) * np.dot(delta_vec[0].T, X )     ]
        for i in range(len(Ts)-1):
            Theta_grads.append( (1.0/m)*np.dot( delta_vec[i+1].T,
                                                np.column_stack( (np.ones(np.shape(zvec[i])[0]), self.actFns[i](zvec[i]))) )
                                                )
        for i in range(len(Ts)-1):
            Theta_grads[i] += (regParams[i]/m)*np.column_stack( ((np.zeros(np.shape(Ts[i])[0])),
                                                                Ts[i][:,1:]))
        return np.array(Theta_grads)
    
    def regGradientWithCost_LogLoss(self, Ts, X, y, regParams):
        J = self.regCost(Ts, X, y, regParams)
        Theta_grads = self.regGradient(Ts, X, y, regParams)
        return (J, Theta_grads)
        
    
    def train(self, tolerance=0.00000001, maxNumIts = 1000, regParams=[1.0,1.0], trainToMax=False, **args):
        print 'Training'
        numIt = -1
        alpha = 0.3   ## TODO - configurable
        costList = []
        while numIt < maxNumIts:
            numIt += 1
            J,G = self.regGradientWithCost(Ts = self.Thetas, X=self.trainData[0], y=self.trainData[1], regParams=regParams)
            for i in range(len(self.Thetas)):
                self.Thetas[i] -= alpha*G[i]
            costList.append(J)
            print 'Iteration %d: Cost = %f' % (numIt, J)
            if numIt > 30:
                percentTage = (costList[numIt-1]-costList[numIt])/J*100.0
                #print 'Decreased by %f percent' % percentTage
                if (not trainToMax) and percentTage < tolerance:
                    break
        print 'Final Log Loss Score (Normalized)'
        print self.ReportLogLossScore()
    
    def unpackTheta(self, Theta1D):
        Ts = []
        start = 0
        for i in range(self.numLayers):
            _shape = np.shape(self.Thetas[i])
            _numEl = _shape[0]*_shape[1]
            #print _shape
            #print _numEl
            #print len(Theta1D)
            #print len(Theta1D[start:(start + _numEl)])
            Ts.append(
                      np.reshape(Theta1D[start:start + _numEl], _shape )
                      )
            start += _numEl
        return Ts
    
    def packThetas(self, T_list):
        #return np.concatenate(  (np.ndarray.flatten(Theta1), np.ndarray.flatten(Theta2)) , axis = 0)

        return np.concatenate( [np.ndarray.flatten(Ti) for Ti in T_list] , axis=0)
    
    def regGradientWithCost_1D(self, T1D, X, y, regParams):
        res = self.regGradientWithCost(self.unpackTheta(T1D), X, y, regParams)
        return (res[0], self.packThetas(res[1]))
    
    def train_cg(self, regParams=[0.01]*10):
        print 'Training with Conjugate Gradient'
        Thetas_expanded = self.packThetas(self.Thetas)
        #print Thetas_expanded
        # Debug here

        print "no error? - norm looks okay"
        print type(self.Thetas)
        ''' TODO - see if packing and unpacking gives us the same Thetas'''
        X = self.trainData[0]
        y = self.trainData[1]
        J, Thetas_expanded = fmincg(lambda(T1D) : self.regGradientWithCost_1D(T1D, X, y, regParams),
                                         Thetas_expanded, MaxIter=200)
        self.Thetas = self.packThetas(Thetas_expanded)
        
        print self.ReportLogLossScore()
    
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


    def LogLossScore_UnNormalized(self, H, y):
        return np.sum(- np.multiply(y, np.log(H)) + np.multiply( y-1, np.log(1.0-H)) )/(np.shape(H)[0]*1.0)
        

    def ReportLogLossScore(self):
        H = self.hypothesis(self.trainData[0])
        y = self.trainData[1]
        for i in xrange(np.shape(H)[0]):
            #print 'i=%d. Sum of probability = %f. Max = %f. Length = %d' % (i, np.sum(H[i]), np.max(H[i]), len(H[i]))
            H[i] /= np.sum(H[i])
        return self.LogLossScore_UnNormalized(H, y)
        

    def test_loadSampleThetas(self):
        import os
        import inspect
        current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        Theta1 = np.loadtxt(os.path.join( current_folder_path,
                                          '../../data/digits/digits_Theta1.txt'), delimiter=',')
        Theta2 = np.loadtxt(os.path.join( current_folder_path, 
                                          '../../data/digits/digits_Theta2.txt'), delimiter=',')
        self.Thetas = [Theta1, Theta2]
    
    def testGradientFiniteDiff(self):
        Theta1, Theta2 = self.Thetas
        nRow1, nCol1 = np.shape(Theta1)
        nRow2, nCol2 = np.shape(Theta2)
        regParams = [4.0]*2
        DEL = 0.000001
        diff_error = 0.0
#        for r in range(nRow1):
#            for c in range(nCol1):
#                print '(r,c)=(%d,%d)' % (r,c)
#                J_before, G_before = self.regGradientWithCost(self.Thetas, X=self.trainData[0], y=self.trainData[1], regParams=regParams)
#                self.Thetas[0][r,c] += DEL
#                J_after, G_after = self.regGradientWithCost(self.Thetas, X=self.trainData[0], y=self.trainData[1], regParams=regParams)
#                #G_before[0][r,c] += (J_after-J_before)/DEL
#                diff_error += np.linalg.norm(G_before[0]-G_after[0]) + np.linalg.norm(G_before[1]-G_after[1])
#                print G_before[0][r,c]
#                print G_after[0][r,c]
#                print (J_after-J_before)/DEL
#                print 'Accumulated Error in theta 1 = ', diff_error
        for r in range(nRow2):
            for c in range(nCol2):
                print '(r,c)=(%d,%d)' % (r,c)
                J_before, G_before = self.regGradientWithCost(self.Thetas, X=self.trainData[0], y=self.trainData[1], regParams=regParams)
                self.Thetas[1][r,c] += DEL
                J_after, G_after = self.regGradientWithCost(self.Thetas, X=self.trainData[0], y=self.trainData[1], regParams=regParams)
                #G_before[0][r,c] += (J_after-J_before)/DEL
                diff_error += np.linalg.norm(G_before[0]-G_after[0]) + np.linalg.norm(G_before[1]-G_after[1])
                print G_before[1][r,c]
                print G_after[1][r,c]
                print (J_after-J_before)/DEL
                print 'Accumulated Error in theta 2= ', diff_error
        
        
def testNeuralNet():
    nn = NeuralNet(trainingData = 'DIGITS',
                   hiddenLayersSize=[3],
                   activationFunctions=['sigmoid']*2)
    print [np.shape(ob) for ob in nn.Thetas]
    #nn.testGradientFiniteDiff()
    nn.train_cg()

def main():
    nn = NeuralNet(trainingData = 'DIGITS',
                   hiddenLayersSize=[200]*2, 
                 activationFunctions=['sigmoid']*3)
    print [np.shape(ob) for ob in nn.Thetas]
    #nn.train(maxNumIts=10000,regParams=[0.1]*3)
    nn.train_cg(regParams=[0.1]*3)
    print [np.shape(i) for i in nn.trainData]
    #nn.test_loadSampleThetas()
    #print(nn.trainData[1] )
    #print(nn.classify(nn.trainData[0]))
    print np.sum((nn.data_labels != nn.classify(nn.trainData[0])))/(1.0*nn.numInputs)

    
if __name__ == "__main__":
    testNeuralNet()
    #main()