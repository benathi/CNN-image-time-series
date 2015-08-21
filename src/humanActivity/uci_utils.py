import numpy as np
import theano
import matplotlib.pyplot as plt
import pickle,os,sys
from ClassifiersOnActivations import getRawData, findActivations
from ggplot import *
import pandas as pd
A_X=0; A_Y=1; A_Z=2; W_X=3; W_Y=4; W_Z=5;
names = ['X Acceleration', 'Y Acceleration', 'Z Acceleration', 'X Angular Acc', 'Y Angular Acc', 'Z Angular Acc']
MAX_TIME = 128
activity_types = ['WALKING','WALKING_UPSTAIRS','WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']


def plotAccelerations(x, acceleration_types = [A_X,A_Y,A_Z]):
    #acceleration_types = [A_X, A_Y, A_Z, W_X, W_Y, W_Z]
    # assume x is one point of shape 0,1,c
    x = np.swapaxes(np.copy(x),0,2)
    print 'current shape of little x', x.shape
    accelerations = pd.DataFrame({'t': range(MAX_TIME)*len(acceleration_types), 
                       'acceleration': np.reshape(x[np.array(acceleration_types)], (-1) ), 
                       'type':sum([[j for i in range(MAX_TIME)] for j in acceleration_types],[])
                                 })
    _plot = ggplot(aes(x='t', y='acceleration', color='type'), data=accelerations) + geom_point() + geom_line()
    print _plot
    return _plot

def plotAllAccelerations(x):
    plotAccelerations(x, acceleration_types=[A_X,A_Y,A_Z]);
    plotAccelerations(x, acceleration_types=[W_X,W_Y,W_Z]);

def formatInput(X):
	X_new = np.reshape(X, (X.shape[0], 1, MAX_TIME, 6))
	return X_new


# makes the plots appear as embedded in the notebook
# construct a function to delay acceleration in time axis or y axis
def delayTime(X, delay_factor=1.20):
    # expect X to be of shape (0,1,c)
    _, max_time, num_ch = X.shape
    newX = np.zeros_like(X)
    #for ch in xrange(num_ch):
    # use inverse mapping. Look up existing method
    for j in xrange(max_time):
        newX[0,j,:] = X[0,max(0, min(max_time-1, int(j/delay_factor))),:]
    return newX

def delayTimeBatch(X, delay_factor=1.20):
    newX = np.zeros_like(X)
    for b in xrange(X.shape[0]):
        newX[b] = delayTime(X[b], delay_factor)
    return newX


def vScaleAcceleration(X, factor=0.8):
    return X*factor

def horizontalShift(x, shift=5):
    # if shift right, pad with value at the left end. vice versa for shift left
    # positive shift is to the left, negative shift is to the right
    _, max_time, num_ch = x.shape
    newX = np.zeros_like(x)
    for j in xrange(max_time):
        newX[0,j,:] = x[0, max(0, min(max_time-1, (j+shift) )), :]
    return newX

def horizontalShiftBatch(X, shift=5):
    newX = np.zeros_like(X)
    for b in xrange(X.shape[0]):
        newX[b] = horizontalShift(X[b], shift)
    return newX