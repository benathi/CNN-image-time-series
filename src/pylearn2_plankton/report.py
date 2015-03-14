'''
Created on Mar 14, 2015

@author: Ben Athiwaratkun (pa338)

'''
#from __future__ import division
import numpy as np
from pylearn2.utils import serial
import theano, os
import theano.tensor as T
from pylearn2.scripts.dbm.show_reconstructions import load_dataset
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
maxPixel = 28

def loadReportData():
    from skimage.io import imread
    from skimage.transform import resize
    import re, glob
    print 'Loading Plankton Testing Data'
    imageSize = maxPixel*maxPixel
    dataPath = '../../data/plankton/test'
    m = 0
    for filename in glob.glob(os.path.join(dataPath, '*.jpg')):
        if filename[-4:] != ".jpg":
            continue
        m += 1
    
    print 'Number of Inputs = %d' % (m)
    XReport = np.empty((m, 1, maxPixel, maxPixel))
    Y_infoName = []
    i = -1
    for filename in glob.glob(os.path.join(dataPath, '*.jpg')):
        if filename[-4:] != ".jpg":
            continue
        i += 1
        image = imread(filename, as_grey=True)
        image = resize(image, (maxPixel, maxPixel))
        XReport[i,0] = image
        shortFileName = re.match(r'.*/(.*)', filename).group(1)
        #print shortFileName
        Y_infoName.append( (shortFileName) )
        break 
        ''' ------------- DEBUG -------------- '''
    XReport = 1.0 - XReport
    Y_infoName = np.array(Y_infoName)
    print 'Shape of X', np.shape(XReport)
    print 'Done Loading Plankton Data'
    return (XReport, Y_infoName, m)

def loadInfoForReport():
    import pickle, inspect
    print 'Loading Info For Report'
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    _ , _, _, classDict =  pickle.load( open(os.path.join(current_folder_path,
                                                          '../../data/planktonTrain.p'), 'rb'))
    XReport, Y_info, numInputs = loadReportData()
    return (classDict, XReport, Y_info, numInputs)

def generateSubmissionFile(model_path, probMatrix, class_names, Y_info):
    import csv
    print 'Generating Submission File'
    headerArray = ['image']
    for i in range(121):
        headerArray.append(class_names[i])
    csvFileName = model_path + '.csv'

    with open(csvFileName, 'wb') as csvfile:
        writeResults = csv.writer(csvfile, delimiter=',')
        writeResults.writerow(headerArray)
        for i, filename in enumerate(Y_info):
            #probMatrix[i] /= np.sum(probMatrix[i]) # This sum is already 1
            probTemp = np.concatenate(([filename],probMatrix[i]),axis = 0)
            writeResults.writerow(probTemp)
    print 'Finished Generating Submission file to ', csvFileName

def MakeCsv3(model_path='plankton_conv_maxout_model.pkl'):
    print 'Loading Model'
    from planktonDataPylearn2 import PlanktonData
    model = serial.load(model_path)
    
    print 'Model input space is ', model.get_input_space()
    
    ds = PlanktonData(which_set='report')
    XReport = np.reshape(ds.get_data(), (ds.get_num_examples(),1,maxPixel, maxPixel))
    m = np.shape(XReport)[0]
    # TODO - Use the model's input space to do the conversion
    XReport = np.swapaxes(XReport,1,2)
    XReport = np.swapaxes(XReport,2,3)
    print 'XReport type = {}. Dimension = {}'.format(type(XReport), np.shape(XReport))
    
    probMatrix = np.zeros((m, 121))
    batch_size = 500
    # model.set_batch_size(batch_size)
    num_extra = m % batch_size
    floor_numBatches = m / batch_size
    print 'Number of Inputs =', m
    print 'Number of Batches =', floor_numBatches
    print 'Num extra = ', num_extra
    for batchIndex in range(floor_numBatches):
        print 'Batch ', batchIndex
        probMatrix[batchIndex*batch_size:(batchIndex+1)*batch_size] = \
            model.fprop(theano.shared(XReport[batchIndex*batch_size:(batchIndex+1)*batch_size],
                                   name='XReport')).eval()
    if num_extra > 0:
        print 'Extra Batch'
        lastX = np.zeros((batch_size, np.shape(XReport)[1], np.shape(XReport)[2], np.shape(XReport)[3]))
        lastX[:num_extra] = XReport[floor_numBatches*batch_size:]
        probMatrix[floor_numBatches*batch_size:] = \
            model.fprop(theano.shared(lastX,name='XReport')).eval()[:num_extra]

    print type(probMatrix)
    print np.shape(probMatrix)
    
    class_names = PlanktonData(which_set='train').class_names
    Y_info = ds.Y_infoName
    generateSubmissionFile(model_path, probMatrix, class_names, Y_info)

def MakeCsv2(model_path='plankton_conv_maxout_model.pkl'):
    print 'Loading Model'
    from planktonDataPylearn2 import PlanktonData
    model = serial.load(model_path)
    input_space = model.get_input_space()
    input_source = model.get_input_source()
    print 'input space', input_space
    print 'input source', input_source
    print type(input_source)
    from pylearn2 import space
    
    
    class_names = PlanktonData(which_set='train').class_names
    ds = PlanktonData(which_set='report')
    #ds = load_dataset('report')
    #XReport = np.reshape(ds.get_data(), (ds.get_num_examples(), 1, 28,28) )
    #print type(XReport)
    #XReport = theano.shared(XReport, name='inputs')
    #XReport.shape(input_space)
    #XReport.reshape(input_space)
    tBatch =  input_space.make_theano_batch()
    print type(tBatch)
    print tBatch
    
    Y_info = ds.Y_infoName
    probMatrix = model.fprop(tBatch)
    #generateSubmissionFile(model_path, probMatrix, class_names, Y_info)

def MakeCsv(model_path='plankton_conv_maxout_model.pkl'):
    print 'Loading Model'
    model = serial.load(model_path)
    print 'Loading Report Data'
    from planktonDataPylearn2 import PlanktonData  
    dataset = PlanktonData('report')
    
    print 'Adding padding to data'
    batch_size = 100
    model.set_batch_size(batch_size)
    m = dataset.X.shape[0]
    print 'Number of inputs size', m
    extra = batch_size - m % batch_size
    assert (m + extra) % batch_size == 0
    if extra > 0:
        dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
        dtype=dataset.X.dtype)), axis=0)
    assert dataset.X.shape[0] % batch_size == 0
    
    
    X = model.get_input_space().make_batch_theano()
    Y = model.fprop(X)
    y = T.argmax(Y, axis=1)
    from theano import function
    f  = function([X], y) # change y to Y?
    
    y = []
    for i in xrange(dataset.X.shape[0] / batch_size):
        x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
        if X.ndim > 2:
            x_arg = dataset.get_topological_view(x_arg)
        y.append(f(x_arg.astype(X.dtype)))
    
    y = np.concatenate(y)

def main():
    # TODO: Take an argument as model filename
    MakeCsv3('plankton_conv_model.pkl')
    
if __name__ == "__main__":
    main()