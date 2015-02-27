'''
Directly imported from other project -- modify
'''
from __future__ import division
import numpy as np
import collections
import random
import math
THETA1_NUM_ROW = 25
THETA1_NUM_COL = 401
THETA2_NUM_ROW = 10
THETA2_NUM_COL = 26
NUM_LABELS = 10
LEN_THETA1 = THETA1_NUM_ROW*THETA1_NUM_COL
LEN_THETA2 = THETA2_NUM_ROW*THETA2_NUM_COL
LEN_THETA = LEN_THETA1 + LEN_THETA2
EPSILON = 0.125



def randomizeMember(epsilon=EPSILON):
    # This method initializes the weights of Theta1, Theta2
    Theta1 = __randomizeParameters(THETA1_NUM_ROW, THETA1_NUM_COL, epsilon)
    Theta2 = __randomizeParameters(THETA2_NUM_ROW, THETA2_NUM_COL, epsilon)
    return packThetas(Theta1, Theta2)
    
def __randomizeParameters(dim_in, dim_out, epsilon):
    # ranges from [-epsilon, epsilon]
    Mat = np.array([ [random.random()*2*epsilon-epsilon for i in xrange(dim_in)] for j in xrange(dim_out)] )
    return Mat

def randomizeMatrixThetas(numTrials):
    Z = np.empty((numTrials,LEN_THETA));
    for i in range(numTrials):
        Z[i] = randomizeMember()
    return Z

def unpackTheta(Theta):
    # tested: passed
    if len(Theta) != LEN_THETA:
        print "Error: The Length of Theta is %d" % len(Theta)
    Theta1 = np.reshape(Theta[:LEN_THETA1], (THETA1_NUM_ROW, THETA1_NUM_COL) )
    Theta2 = np.reshape(Theta[LEN_THETA1:], (THETA2_NUM_ROW, THETA2_NUM_COL) )
    return (Theta1, Theta2)
    
def packThetas(Theta1, Theta2):
    # tested: passed
    return np.concatenate(  (np.ndarray.flatten(Theta1), np.ndarray.flatten(Theta2)) , axis = 0)


def loadData():
    # tested : passed - this code runs from any package name
    import os
    import inspect
    #print os.getcwd()
    #print inspect.getfile(inspect.currentframe())
    #print inspect.getsourcefile(inspect.currentframe())
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    designMatrix = np.loadtxt(os.path.join( current_folder_path, '../Data/digits_designMatrix.txt') )
    data_labels = np.loadtxt(os.path.join( current_folder_path, '../Data/digits_label.txt'))
    Data = collections.namedtuple('Data', ['designMatrix','labels'])
    d = Data(designMatrix, data_labels)
    return d


def neuralNet_Cost(digitsData, Theta, regularized=True, regParam=1):
    # tested: correct
    # cost = 0.287629 for the sample Theta1, Theta2
    designMatrix, data_labels = digitsData
    Theta1, Theta2 = unpackTheta(Theta)
    numDataPoints = np.shape(designMatrix)[0]
    htheta = neuralNet_Predict(Theta1, Theta2, designMatrix)
    expandedY = np.zeros((numDataPoints, NUM_LABELS))
    for row in xrange(numDataPoints):
        _label = data_labels[row] - 1 # 9,0,1,...,8 instead of 10,1,2,..,9
        expandedY[row,_label] = 1.0
    part1 = -np.multiply(expandedY, np.log(htheta))
    part2 = np.multiply(expandedY-1, np.log(1-htheta))
    
    cost = (1/numDataPoints)*np.dot( np.ones((1,numDataPoints)), np.dot( part1 + part2  , np.ones((NUM_LABELS,1))  )   )
    if regularized:
        cost += (regParam/(2*numDataPoints))*( np.sum(np.multiply(Theta1[:,1:],Theta1[:,1:])) + 
                                               np.sum(np.multiply(Theta2[:,1:],Theta2[:,1:])))
    
    return cost[0,0]


def addOneFirstColumn(X):
    X_one = np.empty( (np.shape(X)[0], np.shape(X)[1]+1) )
    X_one[:,1:] = X
    X_one[:,0] = np.ones((np.shape(X)[0]))
    return X_one

def neuralNet_Predict(Theta1, Theta2, X):
    # tested: passed
    # Theta1, Theta2 are the weight parameters
    # X is the data whose labels we want to predict
    #numDataPoints = np.shape(X)[0]
    #numLabels = np.shape(Theta2)[0]
    X_one = addOneFirstColumn(X)
    h1 = sigmoid( np.dot(X_one, Theta1.T ) )
    h1_one = addOneFirstColumn(h1)
    h2 = sigmoid(np.dot(h1_one, Theta2.T))
    return h2
    
def neuralNet_PredictLabel(Theta1, Theta2, X):
    # tested: passed
    h2 = neuralNet_Predict(Theta1, Theta2, X)
    return  (1+ np.argmax(h2,axis=1))
    
def neuralNet_PredictLabel2(Theta, X):
    Theta1, Theta2 = unpackTheta(Theta)
    return neuralNet_PredictLabel(Theta1, Theta2, X)


def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def loadSampleThetas():
    import os
    import inspect
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    Theta1 = np.loadtxt(os.path.join( current_folder_path, '../Data/digits_Theta1.txt'), delimiter=',')
    Theta2 = np.loadtxt(os.path.join( current_folder_path, '../Data/digits_Theta2.txt'), delimiter=',')
    return (Theta1, Theta2)

# This method is used in SA
def neuralNet_NeighborSA(Theta):
    n = len(Theta)
    indexToPerturb = random.randint(0,n-1)
    
    perturbation = random.normalvariate(0, 3*stdOfParameters(indexToPerturb, Theta))
    # do a deep copy
    perturbationMatrix = np.zeros(np.shape(Theta))
    perturbationMatrix[indexToPerturb] = perturbation
    return (Theta + perturbationMatrix)



def neuralNet_Neighbors(Theta, option, **kwargs):
    # This neighbor function perturbs Theta at one point
    # The following options are available
    # 1. perturbs according to the scale epsilon where we assume the 
    # 2. perturbs according to the true scale (obtained by computing the average absolute values)
    Theta_new = np.array(Theta)
    n = len(Theta)
    indexToPerturb = random.randint(0,n-1)
    if option == 'Constant':
        alpha = kwargs['alpha']
        std = alpha*EPSILON
        #print '\tStandard Deviation for Perturbation = %f' % std
        Theta_new[indexToPerturb] += random.normalvariate(0, std)
    elif option == 'DynamicScale':
        alpha = kwargs['alpha']
        std = alpha*aveAbsOfParameters(indexToPerturb,Theta)
        #print '\tStandard Deviation for Perturbation = %f' % std
        Theta_new[indexToPerturb] += random.normalvariate(0, std)
    elif option == 'DynamicStd':
        alpha = kwargs['alpha']
        std = alpha*stdOfParameters(indexToPerturb, Theta)
        Theta_new[indexToPerturb] += random.normalvariate(0, std)
    elif option == 'ConstantRateDecline':
        alpha_initial = kwargs['alpha']
        alpha_final = kwargs['alpha_final']
        current_alpha = (alpha_initial - alpha_final)*(kwargs['maxIteration'] - kwargs['iteration'])/kwargs['maxIteration'] + alpha_final
        std = current_alpha*EPSILON
        Theta_new[indexToPerturb] += random.normalvariate(0, std)
    elif 'RowWise' in option:
        ##### Larger Neighborhood ##################
        theta_index = random.randint(1,2)
        Theta1_new, Theta2_new = unpackTheta(Theta_new)
        #### Setting Alpha
        if option == 'RowWiseConstant':
            alpha = kwargs['alpha']
            std = alpha*EPSILON
        elif option == 'RowWiseConstantRateDecline':
            alpha_initial = kwargs['alpha']
            alpha_final = kwargs['alpha_final']
            current_alpha = (alpha_initial - alpha_final)*(kwargs['maxIteration'] - kwargs['iteration'])/kwargs['maxIteration'] + alpha_final
            std = current_alpha*EPSILON
        #### Do Row-Wise Perturbation
        if theta_index == 1:
            row_to_perturb = random.randint(0,THETA1_NUM_ROW-1)
            for i in range(THETA1_NUM_COL):
                Theta1_new[row_to_perturb, i] += random.normalvariate(0, std)
        else:
            row_to_perturb = random.randint(0,THETA2_NUM_ROW-1)
            for i in range(THETA2_NUM_COL):
                Theta2_new[row_to_perturb, i] += random.normalvariate(0, std)
                
        
        Theta_new = packThetas(Theta1_new, Theta2_new)
    elif 'ColumnWise' in option:
        theta_index = random.randint(1,2)
        Theta1_new, Theta2_new = unpackTheta(Theta_new)
        ### Setting Alpha
        if option == 'ColumnWiseConstant':
            alpha = kwargs['alpha']
            std = alpha*EPSILON
        elif option == 'ColumnWiseConstantRateDecline':
            alpha_initial = kwargs['alpha']
            alpha_final = kwargs['alpha_final']
            current_alpha = (alpha_initial - alpha_final)*(kwargs['maxIteration'] - kwargs['iteration'])/kwargs['maxIteration'] + alpha_final
            std = current_alpha*EPSILON
        ### Do Column-Wise Perturbation
        if theta_index == 1:
            col_to_perturb = random.randint(0,THETA1_NUM_COL-1)
            for i in range(THETA1_NUM_ROW):
                Theta1_new[i, col_to_perturb] += random.normalvariate(0, std)
        else:
            col_to_perturb = random.randint(0,THETA2_NUM_COL-1)
            for i in range(THETA2_NUM_ROW):
                Theta2_new[i, col_to_perturb] += random.normalvariate(0, std)
        Theta_new = packThetas(Theta1_new, Theta2_new)
    ############################
    #print '\tOption=%r \tStandard Deviation for Perturbation = %f' % (option, std)
    return Theta_new



def isTheta1Index(index):
    if index < LEN_THETA1 and index >= 0:
        return True
    else:
        return False
    
def stdOfParameters(index, Theta):
    if isTheta1Index(index):
        return np.std(Theta[0:LEN_THETA1])
    else:
        return np.std(Theta[LEN_THETA1:])

def aveAbsOfParameters(index, Theta):
    if isTheta1Index(index):
        return np.mean(abs(Theta[:LEN_THETA1]))
    else:
        return np.mean(abs(Theta[LEN_THETA1:]))

def main():
    digitsData = loadData()
    #print designMatrix
    #print data_labels
    designMatrix, data_labels = digitsData
    print sigmoid(1)
    Theta1, Theta2 = loadSampleThetas()
    h2 = neuralNet_PredictLabel(Theta1, Theta2, designMatrix)
    # test randomize member
    print neuralNet_Cost(digitsData, packThetas(Theta1, Theta2))
    print neuralNet_Cost(digitsData, randomizeMember())
    print Theta1
    print Theta2
    print designMatrix
    
if __name__ == "__main__":
    main()