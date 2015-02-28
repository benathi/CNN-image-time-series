'''
Created on Feb 27, 2015

@author: keegankang
'''
import numpy as np
import math
''' This function returns the gradient matrix'''
''' Inputs:
X
y
thetaVecs should be a list of thetas for each layer 
size of thetaVecs -1 would be number of hidden layers (last theta is for output)
actFns should be of num hidden layers + 1 
K should be number of labels
'''
def sigmoid(z):
    # Return sigmoid of any input
    return 1/(1+math.exp(-z))
    

def sigmoidDer(z):
    return sigmoid(z) * (1 - sigmoid(z))
    

def gradient(X,y, thetaList, Lambda, actFns=[sigmoid, sigmoid], activationDers=[sigmoidDer, sigmoidDer]):
    pass


if __name__ == "__main__":
    pass
    #print gradient(X,y)



## To remove later once we have inputs

actFns=[sigmoid, sigmoid]
theta1 = np.ones((25,401))
theta2 = np.ones((10,26)) * 0.01
X = np.ones((5000,401)) * 0.02
K = 10
thetaList = [theta1,theta2]
Lambda = 1
y = np.ones((5000,10))

## Calculate Cost Function here
## First calculate h(x)


## To do so, loop through the activation Fns
## Perform matrix multiplication on X^T

## Initial w/o any hidden layers
H = - np.mat(X) * np.mat(thetaList[0].T)
H = np.vectorize(actFns[0])(H)

## Now for each hidden layer:
for i in range(1, len(actFns)):
    H = - np.mat(np.concatenate((np.ones((H.shape[0],1)), H), axis=1))  * np.mat(thetaList[i].T)
    H = np.vectorize(actFns[i])(H)

##  Unregularized Cost Function
J = np.sum(-np.multiply(y, np.log(H)) + np.multiply(1-y, np.log(1-H)))/X.shape[0]

## Regularized Cost Function
penalty = 0


for i in range(len(thetaList)):
    penalty += np.sum(np.multiply(thetaList[i][:,1:],thetaList[i][:,1:]) )

J += Lambda/(len(thetaList)*X.shape[0]) * penalty

print(J)

## Back Propagation
## Hidden layers of the form
## Input ----> HL1 -----> HL2 ---> Output
##         t1        t2        t3
## Thus for L hidden layers, we must have L+1 thetas
## and L+1 a_is, zs, and deltas
## Create a list of matrices to store as, zs, and deltas









