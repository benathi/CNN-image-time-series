'''
This is a wrapper for plankton data
'''
import os, inspect, pickle
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class PlanktonData(DenseDesignMatrix):
    def __init__(self):
        # 0. Loading data
        current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        X , Y, _, classDict =  \
        pickle.load( open(os.path.join(current_folder_path, 
                               '../../data/planktonTrain.p'), 'rb'))   
        # 1. Set class names
        numClasses = 121  
        classNames = []
        for i in range(numClasses):
            classNames.append(classDict[i][0])
        self.class_names = classNames
        
        # 2. Import data 
        m = np.shape(X)[0]
        print 'Input Size = ', m
        numPixels = 28*28
        # Create train/validation/test sets
        randPermutation = np.random.permutation(m)
        X = X[randPermutation,:numPixels]
        X = 1.0 - X
        Y = Y[randPermutation]
        Y_oneHot = np.zeros((np.shape(X)[0], numClasses))
        for i, val in enumerate(Y):
            Y_oneHot[i,val] = 1.0
        super(PlanktonData, self).__init__(X=X, y=Y_oneHot)
# The pixel values are already from 0 to 1    