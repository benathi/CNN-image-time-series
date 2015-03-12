'''
Created on Mar 6, 2015

@author: Ben Athiwaratkun (pa338)

'''
#from __future__ import division
#Import libraries for doing image analysis
from skimage.io import imread
from skimage.transform import resize
import glob
import os
import numpy as np
#import pandas as pd
#from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")
import inspect
import re
import pickle

# get the classnames from the directory structure


#for i in  list(set(glob.glob(os.path.join('/Users/ben/Kaggle/Plankton/Data',"train", "*")))):
#    print i

def loadData():
    print 'Loading Plankton Data'
    maxPixel = 28
    imageSize = maxPixel*maxPixel
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dataPath = '../../data/plankton'
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
    Y_info = []
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
            Y_info.append( (shortFileName, classIndex, className) )
        labelMapText[classIndex] = (className, numInstancesPerClass)
    Y_info = np.array(Y_info)
    print 'Done Loading Plankton Data'
    print 'Dumping Data to Pickle File'
    ''' 
    Y is a label in number
    Y_info is a dictionary mapping index to input information: filename, classIndex, className
    labelMapText maps a class index to class name (text)
    '''
    pickle.dump( (X, Y, Y_info, labelMapText ), open(os.path.join(current_folder_path, '../../data/planktonTrain.p'), 'wb'))
    print 'Done dumping to pickle file'

def loadDataSplitted_LeNetFormat():
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    X , Y, Y_info, classDict =  \
    pickle.load( open(os.path.join(current_folder_path,
                                           '../../data/planktonTrain.p'), 'rb'))
    X = 255.0 - X
    X /= 255.0
    m = np.shape(X)[0]
    print 'Input Size = ', m
    
    numPixels = 28*28
    # Create train/validation/test sets
    randPermutation = np.random.permutation(m)
    X = X[randPermutation,:28*28]
    Y = Y[randPermutation]
    Y_info = Y_info[randPermutation]
    
    train_factor = 0.60
    cv_factor = 0.20
    index_endTrain = (int(train_factor*m)/500)*500
    X_train = X[:index_endTrain]
    Y_train = Y[:index_endTrain]
    index_endCv = (int( (train_factor+cv_factor)*m )/500)*500
    X_cv = X[index_endTrain:index_endCv]
    Y_cv = Y[index_endTrain:index_endCv]
    
    index_endTest = (m/500)*500  # a quick hack
    X_test = X[index_endCv:index_endTest]
    Y_test = Y[index_endCv:index_endTest]
    print 'Done Splitting Data to Train/CV/Test'
    return ( (X_train, Y_train),
             (X_cv, Y_cv),
             (X_test, Y_test) )


def runNeuralNet():
    import NeuralNet
    nn = NeuralNet.NeuralNet(trainingData='PLANKTON',
                             hiddenLayersSize=[59,59],
                 activationFunctions=['sigmoid']*3)
    print [np.shape(ob) for ob in nn.Thetas]
    nn.train(maxNumIts=5000,regParams=[0.01]*3, trainToMax=True)
    #nn.train_cg(regParams=[0.01]*3)

def main():
    loadData()
    #runNeuralNet()
    train, cv, test = loadDataSplitted_LeNetFormat()
    print type(train)
    print np.shape(train[0])
    print np.shape(train[1])
    print np.shape(cv[0])
    print np.shape(cv[1])
    print np.shape(test[0])
    print np.shape(test[1])

if __name__ == "__main__":
    main()