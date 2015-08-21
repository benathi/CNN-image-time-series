'''
Created on Jun 24, 2015

@author: ben

Wrapper for UCI dataset
'''

import os, inspect, pickle
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
NUM_CLASSES = 6

def dataPath(which_set):
    return os.path.join(os.environ['PYLEARN2_DATA_PATH'] , 'human_activity/UCI_HAR_Dataset/' + which_set + '.p')

class UCIData(DenseDesignMatrix):
    def __init__(self, which_set):
        assert which_set in  ['train', 'valid', 'test'], \
        'which_set %r is not in the list %r' % (which_set, ['train', 'valid', 'test'])
        X,Y = pickle.load(open(dataPath(which_set),'rb'))
        Y_oneHot = np.zeros((np.shape(X)[0], NUM_CLASSES))
        for i, val in enumerate(Y):
            # note: val is from 1 to 6
            Y_oneHot[i,val-1] = 1
        # Currently, X has shape b,c,01
        print 'Shape Before swapping axes', X.shape
        X = np.swapaxes(X,1,2)
        print 'Shape after swapping axes', X.shape
        # now, X has shape b, 01, c
        X = np.reshape(X, (X.shape[0], -1)) # this will flatten X which is 3D ('b', 'c', 0) to 2D ('b', 'c'*0)
        super(UCIData, self).__init__(X=X, y=Y_oneHot)

if __name__=='__main__':
    ob = UCIData(which_set='train')
