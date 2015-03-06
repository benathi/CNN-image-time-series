'''
Created on Mar 6, 2015

@author: Ben Athiwaratkun (pa338)

'''
#from __future__ import division
#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier as RF
import glob
import os
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from matplotlib import colors
from pylab import cm
from skimage import segmentation
from skimage.morphology import watershed
from skimage import measure
from skimage import morphology
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import peak_local_max
import warnings
warnings.filterwarnings("ignore")
import inspect
import re
import pickle
import NeuralNet
# get the classnames from the directory structure


#for i in  list(set(glob.glob(os.path.join('/Users/ben/Kaggle/Plankton/Data',"train", "*")))):
#    print i

def loadData():
    print 'Loading Plankton Data'
    maxPixel = 25
    imageSize = maxPixel*maxPixel
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dataPath = '/Users/ben/Kaggle/Plankton/Data'
    directory_names = list(set(glob.glob(os.path.join(dataPath,"train", "*")))\
                           .difference(set(glob.glob(os.path.join(dataPath,"train","*.*")))))
    m = 0
    numClasses = 0
    for directory in directory_names:
        numClasses += 1
        for filename in glob.glob(os.path.join(directory, '*.jpg')):
            if filename[-4:] != ".jpg":
                continue
            m += 1
    print 'Number of Inputs = %d. Number of Classes = %d' % (m, numClasses)
    
    numFeatures = imageSize + 2 # +2 for width and height
    X = np.empty((m, numFeatures))
    Y_dict = {}
    labelMapText = {}
    Y = np.empty((m))
    i = -1
    classIndex = -1
    for directory in directory_names:
        classIndex += 1
        numInstancesPerClass = 0
        className = re.match(r'.*/(.*)', directory).group(1)
        print 'Class #%d\t: %s' %(classIndex, className)
        for filename in glob.glob(os.path.join(directory, '*.jpg')):
            if filename[-4:] != ".jpg":
                continue
            i += 1
            numInstancesPerClass += 1
            image = imread(filename, as_grey=True)
            image = resize(image, (maxPixel, maxPixel))
            X[i,0:imageSize] = np.reshape(image, (1,imageSize))
            X[i,imageSize] = np.shape(image)[0]
            X[i,imageSize+1] = np.shape(image)[1]
            Y[i] = classIndex
            shortFileName = re.match(r'.*/(.*)', filename).group(1)
            Y_dict[i] = (shortFileName, classIndex, className)
        labelMapText[classIndex] = (className, numInstancesPerClass)
    print 'Done Loading Plankton Data'
    pickle.dump( (X, Y, Y_dict, labelMapText ), open(os.path.join(current_folder_path, '../../data/planktonTrain.p'), 'wb'))
    print 'Done dumping to pickle file'

def runNeuralNet():
    nn = NeuralNet.NeuralNet(trainingData='PLANKTON',
                             hiddenLayersSize=[59, 59], 
                 activationFunctions=['sigmoid']*4)
    print [np.shape(ob) for ob in nn.Thetas]
    nn.train(maxNumIts=10,regParams=[0.01]*4,
             trainToMax=True)
    print [np.shape(i) for i in nn.trainData]
    #nn.test_loadSampleThetas()
    #print(nn.trainData[1] )
    #print(nn.classify(nn.trainData[0]))
    print 'Log Loss Score'
    print nn.gradient(regParams=[0.0]*4)[0]

# define the log loss metric

def main():
    loadData()
    runNeuralNet()


if __name__ == "__main__":
    main()