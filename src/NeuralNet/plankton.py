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
import convertImage

# get the classnames from the directory structure


#for i in  list(set(glob.glob(os.path.join('/Users/ben/Kaggle/Plankton/Data',"train", "*")))):
#    print i

def dumpPlanktonData(maxPixel=40, cache=False):
    print 'Loading Plankton Data'
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
            ## Record image size here (28 by 28)
            #X[i,imageSize] = np.shape(image)[0]
            #X[i,imageSize+1] = np.shape(image)[1]
            #image = convertImage.process(image)
            image = resize(image, (maxPixel, maxPixel))
            X[i,0:imageSize] = np.reshape(image, (1,imageSize))
            Y[i] = classIndex
            shortFileName = re.match(r'.*/(.*)', filename).group(1)
            Y_info.append( (shortFileName, classIndex, className) )
        labelMapText[classIndex] = (className, numInstancesPerClass)
    Y_info = np.array(Y_info)
    print 'Done Loading Plankton Data'
    print 'Start Shuffling'
    np.random.seed(15430967)
    randPermutation = np.random.permutation(m)
    Y_info = Y_info[randPermutation]
    X = X[randPermutation]
    Y = Y[randPermutation]
    print 'Dumping Data to Pickle File'
    ''' 
    Y is a label in number
    Y_info is a dictionary mapping index to input information: filename, classIndex, className
    labelMapText maps a class index to class name (text)
    '''
    if cache:
        pickle.dump( (X, Y, Y_info, labelMapText ), open(os.path.join(current_folder_path, '../../data/planktonTrain' + str(maxPixel) +'.p'), 'wb'))
        print 'Done dumping to pickle file'
    else:
        return (X,Y,Y_info, labelMapText)


def loadDataSplitted_LeNetFormat():
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    X , Y, Y_info, classDict =  \
    pickle.load( open(os.path.join(current_folder_path,
                                           '../../data/planktonTrain.p'), 'rb'))
    # The pixel values are already from 0 to 1
    m = np.shape(X)[0]
    print 'Input Size = ', m
    
    numPixels = 28*28
    # Create train/validation/test sets
    randPermutation = np.random.permutation(m)
    X = X[randPermutation,:28*28]
    X = 1.0 - X
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
    
def sanitycheck():
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    X , Y, Y_info, classDict =  \
    pickle.load( open(os.path.join(current_folder_path,
                                           '../../data/planktonTrain.p'), 'rb'))  
    print classDict  
    print Y_info
    print X
    print Y

def loadReportData():
    print 'Loading Plankton Testing Data'
    maxPixel = 28
    imageSize = maxPixel*maxPixel
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    dataPath = '../../data/plankton/test'
    m = 0
    for filename in glob.glob(os.path.join(dataPath, '*.jpg')):
        if filename[-4:] != ".jpg":
            continue
        m += 1
    
    print 'Number of Inputs = %d' % (m)
    numFeatures = imageSize + 2 # +2 for width and height
    XReport = np.empty((m, numFeatures))
    Y_infoName = []
    i = -1
    for filename in glob.glob(os.path.join(dataPath, '*.jpg')):
        if filename[-4:] != ".jpg":
            continue
        i += 1
        image = imread(filename, as_grey=True)
        XReport[i,imageSize] = np.shape(image)[0]
        XReport[i,imageSize+1] = np.shape(image)[1]
        image = resize(image, (maxPixel, maxPixel))
        XReport[i,0:imageSize] = np.reshape(image, (1,imageSize))   
        shortFileName = re.match(r'.*/(.*)', filename).group(1)
        #print shortFileName
        Y_infoName.append( (shortFileName) )
        #print Y_infoName
    XReport = XReport[:28*28]
    XReport = 1.0 - XReport         
    Y_infoName = np.array(Y_infoName)
    print 'Done Loading Plankton Data'
    return (XReport, Y_infoName)
    
    #probMatrix = getProbMatrix(XReport)
    #current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    #_ , _, _, classDict =  \
    #pickle.load( open(os.path.join(current_folder_path,
    #                                       '../../data/planktonTrain.p'), 'rb'))
    #generateSubmissionFile(probMatrix, classDict, Y_infoName)

def getProbMatrix(XReport):
    # TODO - implement
    pass

def generateSubmissionFile(probMatrix, classDict, Y_info):
    import csv
    print 'Generating Submission File'
    headerArray = ['image']
    for i in range(121):
        headerArray.append(classDict[i][0])
    with open('results.csv', 'wb') as csvfile:
        writeResults = csv.writer(csvfile, delimiter=',')
        writeResults.writerow(headerArray)
        for i, filename in enumerate(Y_info):
            probMatrix[i] /= np.mean(probMatrix)
            probTemp = np.concatenate(([filename],probMatrix[i]),axis = 0)
            #print probTemp
            writeResults.writerow(probTemp)
        
    '''
    print headerArray
    for i,filename in enumerate(Y_info):
        print filename
        print i
        # write filename (no new line)
        # write probMatrix[i] (whole line, csv) + newline
    print 'Now appending to Probability Matrix'
    '''
    
    
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-maxpixel', action="store", default=40, type=int)
    allArgs = parser.parse_args()
    dumpPlanktonData(allArgs.maxpixel)
    pass

if __name__ == "__main__":
    main()