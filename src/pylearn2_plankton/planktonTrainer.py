'''
Created on Mar 13, 2015

@author: benathi
'''
from pylearn2.config import yaml_parse
import os,sys,inspect
import theano
import numpy as np


def train():
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    filePath = os.path.join(current_folder_path, 'plankton_conv.yaml')
    print 'Reading YAML Configurations'
    trainObj = open(filePath,'r').read()
    print 'Loading Train Model'
    trainObj = yaml_parse.load(trainObj)
    print 'Looping'
    trainObj.main_loop()
    return trainObj

def loadReportData():
    from skimage.io import imread
    from skimage.transform import resize
    import re, glob
    print 'Loading Plankton Testing Data'
    maxPixel = 28
    imageSize = maxPixel*maxPixel
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

def loadInfoForReport():
    import pickle
    print 'Loading Info For Report'
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    _ , _, _, classDict =  pickle.load( open(os.path.join(current_folder_path,
                                                          '../../data/planktonTrain.p'), 'rb'))
    XReport, Y_info = loadReportData()
    return (classDict, XReport, Y_info)

def trainAndReport():
    trainObj = train()
    classDict, XReport, Y_info = loadInfoForReport()
    print 'Generating Probability Matrix'
    probMatrix = trainObj.fprop(theano.shared(XReport, name='XReport')).eval()
    generateSubmissionFile(probMatrix, classDict, Y_info)

if __name__=='__main__':
    trainAndReport()