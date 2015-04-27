'''
Created on Apr 18, 2015

@author: ben
'''
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

'''
Note: format of Y_train is 
'''
def trainRF(X_train, Y_train, X_test, Y_test, model_name, cache=False, n_estimators=400):
    rf_filename = model_name.split('.')[0] + str('RFmodel.p')
    print 'CNN model name', model_name
    print 'Training RF - RF Model Filename =', rf_filename
    if cache:
        if os.path.isfile(rf_filename):
            print 'Model Exists - Loading from disk'
            return pickle.load(open(rf_filename,'rb'))
    print 'number of estimators =', n_estimators
    rf_clf = RandomForestClassifier(n_estimators=n_estimators
                                    ,criterion='gini' # gini
                                    )
    # try lower n_estimator and max_depth=10 (currently no max depth)
    # criterion choice: entropy versys gini
    rf_clf.fit(X_train, Y_train)
    if cache:
        print 'Saving Model to Disk'
        pickle.dump(rf_clf, open(rf_filename, 'wb'))
        print 'Done Saving Model to Disk'
    predictionScores(rf_clf, X_test, Y_test)

def trainSVM(X_train, Y_train, X_test, Y_test, model_name, cache=False, one_vs_rest=True):
    rf_filename = model_name.split('.')[0] + str('RFmodel.p')
    print 'CNN model name', model_name
    print 'Training SVM -SVM Model Filename =', rf_filename
    if cache:
        if os.path.isfile(rf_filename):
            print 'Model Exists - Loading from disk'
            return pickle.load(open(rf_filename,'rb'))
    if one_vs_rest:
        print 'One Versus Rest SVM'
        svm_clf = svm.LinearSVC()       # one versus rest
    else:
        print 'One Versus One SVM'
        svm_clf = svm.SVC()             # one versus one
    svm_clf.fit(X_train, Y_train)   
    if cache:
        print 'Saving Model to Disk'
        pickle.dump(svm_clf, open(rf_filename, 'wb'))
        print 'Done Saving Model to Disk'
    predictionScores(svm_clf, X_test, Y_test)


'''
    designMatrix_train = np.array(np.reshape(designMatrix, 
        (ds.get_num_examples(), 1, 28, 28) ), dtype=np.float32)
'''

def getRawData(data_spec, which_set, maxPixel):
    print 'Loading Raw Data set', which_set
    #PC = __import__('pylearn2_plankton.planktonDataConsolidated', fromlist=['planktonDataConsolidated'])
    PC = __import__('pylearn2_plankton.planktonDataConsolidated', fromlist=[data_spec.split('.')[1]])
    ds = PC.PlanktonData(which_set, maxPixel)
    designMatrix = ds.get_data()[0] # index 1 is the label
    Y = ds.get_data()[1]
    Y = np.where(Y)[1]
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
    return (rawX_train, Y_train, rawX_test, Y_test)

'''
rf_cl:     random forest classifier as a function
X_test:    design matrix
Y_test:    Does it work for one-hot?
'''
def predictionScores(cl, X_test, Y_test):
    print 'Accuracy Score = ', cl.score(X_test, Y_test)

def rfOnActivationsPerformance(model_name, data_spec, which_layer, maxPixel):
    print '\n-----#####-----#####-----#####-----#####-----#####-----#####'
    print 'Performing Tests using Activations on other classifiers'
    print 'pklname', model_name
    print 'data_spec', data_spec
    print 'which_layer', which_layer
    print 'maxPixel', maxPixel
    print 'Obtaining Activations'
    X_train, Y_train, X_test, Y_test= prepXY(model_name, data_spec, which_layer, maxPixel)
    trainSVM(X_train, Y_train, X_test, Y_test, model_name, one_vs_rest=True)
    trainSVM(X_train, Y_train, X_test, Y_test, model_name, one_vs_rest=False)
    print 'Running Random Forests'
    for i in range(4):
        print '\tRF Trial', i
        trainRF(X_train, Y_train, X_test, Y_test, model_name)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-layer', action="store", default=2, type=int)
    #parser.add_argument('-yaml', action="store", default='plankton_conv_visualize_model.yaml')
    parser.add_argument('-pklname', action="store", default='model_files/plankton_conv_visualize_model_CPU.pkl')
    parser.add_argument('-data', action="store", default='pylearn2_plankton.planktonDataConsolidated')
    parser.add_argument('-maxpix', action="store", default=28, type=int)
    allArgs = parser.parse_args()
    rfOnActivationsPerformance(model_name=allArgs.pklname,
                               data_spec=allArgs.data,
                               which_layer=allArgs.layer,
                               maxPixel=allArgs.maxpix)