'''
Created on Apr 18, 2015

@author: ben
'''
import numpy as np
import pickle, os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm



'''
Note: format of Y_train is 
'''
def trainRF(X_train, Y_train, X_test, Y_test, model_name, cache=False, n_estimators=600):
    print 'CNN model name', model_name
    print 'number of estimators =', n_estimators
    rf_clf = RandomForestClassifier(n_estimators=n_estimators
                                    ,criterion='gini' # gini
                                    )
    # try lower n_estimator and max_depth=10 (currently no max depth)
    rf_clf.fit(X_train, Y_train)
    predictionScores(rf_clf, X_test, Y_test)
    
    ''' Add Feature Importance '''
    importances = rf_clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf_clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    if False:
        for f in range(X_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    print 'Total number of features', len(indices)
    thres = 1/(100.*len(indices))
    print 'Threshold = ', thres
    print 'Number of features with importance > threshold', np.sum(importances > thres)
    

def trainSVM(X_train, Y_train, X_test, Y_test, model_name, cache=False, one_vs_rest=True):
    print 'CNN model name', model_name
    if one_vs_rest:
        print 'One Versus Rest SVM'
        svm_clf = svm.LinearSVC()       # one versus rest
    else:
        print 'One Versus One SVM'
        svm_clf = svm.SVC()             # one versus one
    svm_clf.fit(X_train, Y_train)   
    predictionScores(svm_clf, X_test, Y_test)


def findActivations(model_name, listX_raw, which_layer, maxPixel=128, verbose=False, batch_size=50):
    # 1. load model file
    import theano
    from pylearn2.utils import serial
    model = serial.load(model_name)
    if verbose: print 'Model input space is ', model.get_input_space()
    
    # 2. find activations at that layer
    num_x = len(listX_raw)
    activation_list = []
    for X in listX_raw:
        if verbose: print 'Forward Propagation'
        if verbose: print 'Shape of X Before Swap', X.shape
        m = X.shape[0]
        #X = np.reshape(X, (m,6,maxPixel,1)) # confirms if this behaves the same way as in Pylearn2
        # 'b', 'c', 0, 1
        #X = np.swapaxes(X,1,2)
        #X = np.swapaxes(X,2,3)
        if verbose: print 'XReport type = {}. Dimension = {}'.format(type(X), np.shape(X))
        activation = None
        #batch_size = 1
        num_batches = m/batch_size
        for batchIndex in range(num_batches):
            if verbose: print 'Finished %.2f Percent' % (100*batchIndex/(1.*num_batches))
            _input = np.array(X[batchIndex*batch_size:(batchIndex+1)*batch_size],
                    dtype=theano.config.floatX)
            if verbose and batchIndex == 0: print 'shape of input', _input.shape
            fprop_results = model.fprop(theano.shared(_input,
                            name='XReport'), return_all=True)[which_layer].eval()
            if batchIndex==0:
                if verbose: print 'fprop_results shape', fprop_results.shape
            if activation is None:
                activation = np.empty((m,) + fprop_results.shape[1:])
                activation[0:batch_size,...] = fprop_results
            else:
                activation[batch_size*batchIndex:batch_size*(1+batchIndex),...] = fprop_results
                #activation = np.concatenate((activation, fprop_results), axis=0)
        # need to flatten to be (num points, num features)
        if verbose: print 'Activation shape before reshape', activation.shape
        activation = np.reshape(activation, (m,-1))
        if verbose: print 'After reshape', activation.shape
        activation_list.append(activation)
    return activation_list

'''
    designMatrix_train = np.array(np.reshape(designMatrix, 
        (ds.get_num_examples(), 1, 28, 28) ), dtype=np.float32)
'''

def getRawData(data_spec, which_set, maxPixel=128, debug=True):
    print 'Loading Raw Data set', which_set
    #PC = __import__('train.planktonDataConsolidated', fromlist=['planktonDataConsolidated'])
    #PC = __import__('train.planktonDataConsolidated', fromlist=[data_spec.split('.')[1]])
    PC = __import__(data_spec, fromlist=[data_spec.split('.')[1]])
    #ds = PC.PlanktonData(which_set, maxPixel)
    ds = PC.UCIData(which_set)
    designMatrix = ds.get_data()[0] # index 1 is the label
    Y = ds.get_data()[1]
    if debug: print Y 
    Y = np.where(Y)[1]
    if debug:
    	print 'After np.where'
    	print Y
    return (designMatrix, Y)
    
'''
Return both X (activations) and Y
Note: configured for 28 x 28 and 3 layer model for now
'''
def prepXY(model_name, data_spec, which_layer, maxPixel):
    # 1. get data
    rawX_train, Y_train = getRawData(data_spec, 'train', maxPixel)
    rawX_cv, Y_cv = getRawData(data_spec, 'valid', maxPixel)
    rawX_test, Y_test = getRawData(data_spec, 'test', maxPixel)
    # combine CV to test
    rawX_train = np.concatenate((rawX_train, rawX_cv), axis=0)
    Y_train = np.concatenate((Y_train, Y_cv), axis=0)
    X_train, X_test = findActivations(model_name, [rawX_train, rawX_test], which_layer, maxPixel)
    # 2. find activations
    print 'Done Finding Activations'
    return (X_train, Y_train, X_test, Y_test)

'''
rf_cl:     random forest classifier as a function
X_test:    design matrix
Y_test:    Does it work for one-hot?
'''
def predictionScores(cl, X_test, Y_test):
    #print 'max of y', np.max(Y_test)
    #print 'min of y', np.min(Y_test)
    print 'Accuracy Score = ', cl.score(X_test, Y_test)

def rfOnActivationsPerformance(model_name, data_spec, which_layer, maxPixel):
    print '\n-----#####-----#####-----#####-----#####-----#####-----#####'
    print 'Performing Tests using Activations on other classifiers'
    print 'pklname', model_name
    print 'data_spec', data_spec
    print 'which_layer', which_layer
    print 'maxPixel', maxPixel
    print 'Obtaining Activations'
    ''' Note : using which_layer as the maximum layer instead '''
    #for layer in range(which_layer):
    layer = which_layer
    print '\nSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS'
    print 'Current Layer (from 0)', layer
    X_train, Y_train, X_test, Y_test= prepXY(model_name, data_spec, layer, maxPixel)
    ## this is to test the CNN predictions (for layer = -1 which is the last layer)
    Y_train_predicted = np.argmax(X_train, axis=1)
    print 'shape of X_train', X_train.shape
    print 'shape of Y_train_predicted', Y_train_predicted.shape
    print 'Train Accuracy', np.mean(Y_train == Y_train_predicted)
    print 'max Y_train', np.max(Y_train)
    print 'max Y_train_predicted', np.max(Y_train_predicted)
    Y_test_predicted = np.argmax(X_test, axis=1)
    print 'Test Accuracy', np.mean(Y_test == Y_test_predicted)

    for _i in range(20):
        print 'X_train', X_train[_i]
        print 'Y_train_predicted = %r. Y_train = %r' % (Y_train_predicted[_i], Y_train[_i])
    print '\n---------------------\n'
    for _i in range(20):
        print 'X_test', X_test[_i]
        print 'Y_test_predicted = %r. Y_test = %r' % (Y_test_predicted[_i], Y_test[_i])
    
    
    return

    print 'Running Random Forests'
    for i in range(1):
        print '\tRF Trial', i
        trainRF(X_train, Y_train, X_test, Y_test, model_name)
    trainSVM(X_train, Y_train, X_test, Y_test, model_name, one_vs_rest=True)
    trainSVM(X_train, Y_train, X_test, Y_test, model_name, one_vs_rest=False)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-layer', action="store", default=2, type=int)
    #parser.add_argument('-yaml', action="store", default='plankton_conv_visualize_model.yaml')
    parser.add_argument('-pklname', action="store", default='model_files/uci_har_model7.pkl')
    parser.add_argument('-data', action="store", default='humanActivity.uciData')
    parser.add_argument('-maxpix', action="store", default=128, type=int)
    allArgs = parser.parse_args()
    rfOnActivationsPerformance(model_name=allArgs.pklname,
                               data_spec=allArgs.data,
                               which_layer=allArgs.layer,
                               maxPixel=allArgs.maxpix)
