'''
Created on Apr 12, 2015

@author: ben
'''


import pickle
import os
import numpy as np

'''
Loads a dictionary in the format [classNum: className, ]
'''
def getClassNameDict(which_data='train'):
    if which_data == 'train':
        return np.load(open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'plankton/train_classNameDict.p'), 'r'))


'''
Loads a list of labels
E.g. [0, 2, 0, 1, ...]
0: class 0, 1: class 1: and so on
'''
def getY_label_int(which_data='train'):
    if which_data == 'train':
        return np.load(open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'plankton/train_Y_label.npy'), 'r'))

def dumpParameters():
    _ , Y, _, classDict = pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain.p'), 'rb'))
    classNameDict = {}
    for key in classDict:
        classNameDict[key] = classDict[key][0]
    pickle.dump(classNameDict, open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'plankton/train_classNameDict.p'), 'wb'))
    np.save(open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'plankton/train_Y_label.npy'), 'wb'),
               np.array(Y, dtype=np.int32))

if __name__ == "__main__":
    #dumpParameters()
    print getClassNameDict()
    print getY_label_int()