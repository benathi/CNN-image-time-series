'''
Created on Mar 30, 2015
@author: keegankang
'''

import numpy as np
import os
import inspect
import pickle
import random
import csv
from test.test_global import GlobalTests



'''
Calculate SubDifferential of a Matrix 
'''
def CalcSubDifferential(A,delta):
    n = A.shape[1]
    m = A.shape[0]
    U, D, V = np.linalg.svd(A, full_matrices=True)
    s = sum(D < delta) + n - len(D)
    U1 = U[:,0:(n-s)]
    V1 = V[:,0:(n-s)]    
    
    if sum(D<delta) > 0:
        U2 = U[:,(n-s):min(n,m)]
        V2 = V[:,(n-s):min(n,m)]
        Brow = U2.shape[1]
        Bcol = V2.shape[1]     
        B = np.random.standard_normal(size=(Brow,Bcol))
        D =  np.linalg.svd(B, full_matrices=False, compute_uv = False)
        B = B/sum(D)
        diffA = np.dot(U1, V1.T) + np.dot(np.dot(U2, B), V2.T)
    else:
        diffA = np.dot(U1, V1.T)
    return diffA


'''
Find T ; using batch learning
        dimsize - Dimension you want to reduce to 
        Stepsize 
        gamma = ||T|| (to prevent trivial solution of T = 0)
        delta = delta for sub differential of matrix (analogous to epsilon)
        epsilon = stopping criterion
        numIter 
        numBatches  

'''
def FindT(X,Y, dimsize, numIter = 200, numBatches = 10, stepsize = 0.05, gamma = 1, delta = 0.001, epsilon = 0.001):
    numObs = X.shape[0]
    origDim = X.shape[1]
    
    ## Create arbitrary starting T for each batch ; could explicitly load file here if needed
    T_init = np.random.standard_normal(size=(numBatches,dimsize,origDim))
    
    ## Otherwise, this    
    TbMat = T_init
    
    ## Normalize each T in TbMat
    for i in range(numBatches):
        normTb = sum(np.linalg.svd(TbMat[i,::], full_matrices=False, compute_uv = False))
        TbMat[i,::] = gamma * TbMat[i,::]/normTb
        
    
    ## Create TbMatDiffMat for matrix differential
    TbMatDiffMat = np.zeros(shape=(numBatches,dimsize,origDim))    
    
    ## TbMat and TbDiffMat stores the T matrix for each batch, and each differential of the matrix
    ## They are updated at every iteration
    
    ## Arbitarily set the global T to something
    globalT = TbMat[1,::]
    
    ## Store current minimum
    minT = globalT 
    
    ## Store cost at each iteration
    costvec = np.zeros(shape=(numIter))
    
    # First, create a vector that demarcates each position depending on number of Batches
    # Toy example: If we had 404 obs, and 10 batches, 
    # 404 / 10 = 40 (whole number)
    # 404 * 10 = 4 (left over)
    # 4 batches get 40 + 1
    # remaining 10-4 get 40
    
    numinBatch = numObs/numBatches
    numMore = numObs % numBatches
    
    Batchvec = np.zeros(shape = (numBatches + 1))
    
    if (numObs % numBatches == 0):
        for i in range(numBatches):
            Batchvec[i+1] = (i+1)* numObs/numBatches
        
    else:
        for i in range(numMore):
            Batchvec[i+1] = (i+1) * (numObs/numBatches + 1)
        for i in range(numMore, numBatches):
            Batchvec[i+1] = Batchvec[i] + numObs/numBatches
    
    ## How to read batchvec
    ## Each batch is from index number [i], to (index number [i+1] - 1)
    
    print Batchvec
    
    ## Randomly shuffle Y and X
    tmp = range(numObs)
    random.shuffle(tmp)
    Y = np.array(Y)
    X = X[tmp,:]
    Y = Y[[tmp]]

    totalnumClassvec = np.unique(Y)
    totalnumClass = len(totalnumClassvec)
   
    for i in range(numIter):
        ## Calculate each T_b diff for each batch
        for b in range(numBatches):
            currX = X[Batchvec[b]:(Batchvec[b+1]-1),:]
            currY = Y[Batchvec[b]:(Batchvec[b+1]-1)]
            
            ## Classvec corresponding to classes, maybe [1, 4, 5, 6] 
            numClassvec = np.unique(currY)
            
            ## Total number of classes, in above case would be 4
            numClasses = len(numClassvec)
            
            batchSum = np.zeros(shape = (TbMat.shape[1],TbMat.shape[2]))
            for j in range(numClasses):
                
                ## Here, class index gives the indices of the appropriate class
                ix = np.in1d(currY.ravel(), numClassvec[j])
                tmp = np.where(ix)
                classindex = tmp[0]
                
                batchX = currX[classindex,:]
                batchSum += np.dot(CalcSubDifferential( np.dot(TbMat[b,::], batchX.T) , delta), batchX)
                
            TbMatDiffMat[b,::] = batchSum - np.dot(CalcSubDifferential(np.dot(TbMat[b,::], currX.T), delta),currX)
            
            TbMat[b,::] -= stepsize * TbMatDiffMat[b,::]
            TbMat[b,::] = gamma * TbMat[b,::]/sum(np.linalg.svd(TbMat[b,::], full_matrices=False, compute_uv = False))
        
        globalSum = np.zeros(shape = (TbMat.shape[1],TbMat.shape[2]))
        
        for b in range(numBatches):
            globalSum += TbMatDiffMat[b,::]
        
        globalT -= stepsize * globalSum
        globalT = gamma*globalT/sum(np.linalg.svd(globalT, full_matrices=False, compute_uv = False))
        print "Iteration", i
        
        ## Print Cost Function
        costSum = 0
        for j in range(totalnumClass):
            ix = np.in1d(Y.ravel(), totalnumClassvec[j])
            tmp = np.where(ix)
            classindex = tmp[0]  
            costX = X[classindex,:]
            costSum += sum(np.linalg.svd(np.dot(globalT,costX.T ), full_matrices=False, compute_uv = False))
        
        costvec[i] = costSum - sum(np.linalg.svd(np.dot(globalT,X.T ), full_matrices=False, compute_uv = False))
        
        print costvec[i]
        
        if(i>1):
            if (np.min(costvec[0:(i-1)]) == costvec[i]):
                minT = globalT
            
            if (np.abs(costvec[i] - costvec[i-1]) < epsilon):
                break
        np.savetxt("TMatrx", minT)
        
    return minT, globalT
        
            
                     
        
        
        
                
                
                
    
    
    

def main():
    ## Load stuff here
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    X , Y, Y_info, classDict =  \
    pickle.load( open(os.path.join(current_folder_path,
                                           '../../data/planktonTrain.p'), 'rb'))
    FindT(X,Y,50)
    
if __name__ == "__main__":
    #testNeuralNet()

    main()