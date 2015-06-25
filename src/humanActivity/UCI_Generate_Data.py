'''
Created on Jun 24, 2015

@author: ben
'''
import numpy as np
import os, sys
import pickle

DIM = 128

# This file is to parse UCI Data into the format that's ready for CNN
def dataPath(filename):
    return os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,filename)

def loadUCIdata_train():
    print 'Loading Train Set'
    y_path = dataPath('human_activity/UCI_HAR_Dataset/train/y_train.txt')
    X_path = ['' for i in range(6)]
    X_path[0] = dataPath('human_activity/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_x_train.txt')
    X_path[1] = dataPath('human_activity/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_y_train.txt')
    X_path[2] = dataPath('human_activity/UCI_HAR_Dataset/train/Inertial_Signals/body_acc_z_train.txt')
    X_path[3] = dataPath('human_activity/UCI_HAR_Dataset/train/Inertial_Signals/body_gyro_x_train.txt')
    X_path[4] = dataPath('human_activity/UCI_HAR_Dataset/train/Inertial_Signals/body_gyro_y_train.txt')
    X_path[5] = dataPath('human_activity/UCI_HAR_Dataset/train/Inertial_Signals/body_gyro_z_train.txt')
    Y = np.loadtxt(y_path)
    num_samples = len(Y)
    X = np.empty((num_samples, 6, DIM))
    for i in range(6):
        X[:,i,:] = np.loadtxt(X_path[i])
    print 'X shape', X.shape

    rand_permutation = np.random.permutation(num_samples)
    X = X[rand_permutation]
    Y = Y[rand_permutation]

    X_train = X[0:6300]
    Y_train = Y[0:6300]
    X_valid = X[-1050:]
    Y_valid = Y[-1050:]

    pickle.dump(
        (X_train, Y_train)
        ,open(dataPath('human_activity/UCI_HAR_Dataset/train.p'),'wb')
        )
    pickle.dump(
        (X_valid, Y_valid)
        ,open(dataPath('human_activity/UCI_HAR_Dataset/valid.p'),'wb')
        )
    
    
def loadUCIdata_test():
    print 'Loading Test Set'
    y_path = dataPath('human_activity/UCI_HAR_Dataset/test/y_test.txt')
    X_path = ['' for i in range(6)]
    X_path[0] = dataPath('human_activity/UCI_HAR_Dataset/test/Inertial_Signals/body_acc_x_test.txt')
    X_path[1] = dataPath('human_activity/UCI_HAR_Dataset/test/Inertial_Signals/body_acc_y_test.txt')
    X_path[2] = dataPath('human_activity/UCI_HAR_Dataset/test/Inertial_Signals/body_acc_z_test.txt')
    X_path[3] = dataPath('human_activity/UCI_HAR_Dataset/test/Inertial_Signals/body_gyro_x_test.txt')
    X_path[4] = dataPath('human_activity/UCI_HAR_Dataset/test/Inertial_Signals/body_gyro_y_test.txt')
    X_path[5] = dataPath('human_activity/UCI_HAR_Dataset/test/Inertial_Signals/body_gyro_z_test.txt')
    Y = np.loadtxt(y_path)
    num_samples = len(Y)
    X = np.empty((num_samples, 6, DIM))
    for i in range(6):
        X[:,i,:] = np.loadtxt(X_path[i])
    print 'X shape', X.shape
    # hack: add 4 more duplicates 
    X_test = np.empty((2950,6,128))
    X_test[:num_samples,:] = X
    X_test[num_samples:,:] = X[0:3]
    Y_test = np.empty((2950))
    Y_test[:num_samples] = Y
    Y_test[num_samples:] = Y[0:3]
    pickle.dump(
        (X_test, Y_test)
        ,open(dataPath('human_activity/UCI_HAR_Dataset/test.p'),'wb')
        )

if __name__ == '__main__':
    loadUCIdata_train()
    loadUCIdata_test()