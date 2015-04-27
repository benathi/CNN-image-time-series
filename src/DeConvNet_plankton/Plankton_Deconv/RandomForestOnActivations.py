'''
Created on Apr 18, 2015

@author: ben
'''
import numpy as np
import pickle, os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from plankton_featureMapActivations import findActivations
# Custom Library
'''
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
class_path = parent_folder + '/' + 'utils'
if class_path not in sys.path:
    sys.path.append( class_path )
'''
#import plankton_utils


'''
Note: format of Y_train is 
'''
def trainRF(X_train, Y_train, X_test, Y_test, model_name, cache=False, n_estimators=500):
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
Return both X (activations) and Y
Note: configured for 28 x 28 and 3 layer model for now
'''
def prepXY(data_spec, model_name,  cache=True,
           test_start_idx=29000):
    filename_X = model_name.split('.')[0] +'_X' + '.npy'
    filename_Y = model_name.split('.')[0] +'_Y' + '.npy'
    if cache:
        if os.path.isfile(filename_X) and os.path.isfile(filename_Y):
            X = np.load(open(filename_X, 'r'))
            Y = np.load(open(filename_Y, 'r'))
            X_train = X[:test_start_idx]
            Y_train = Y[:test_start_idx]
            X_test = X[test_start_idx:]
            Y_test = Y[test_start_idx:]
            return (X_train, Y_train, X_test, Y_test)
    # 1. get data
    from pylearn2_plankton.planktonDataPylearn2 import PlanktonData
    ds = PlanktonData(which_set='train') # train includes test set
    designMatrix = ds.get_data()[0] # index 1 is the label
    Y = ds.get_data()[1]
    Y = np.where(Y)[1]
    print 'type of Y_train', type(Y)
    print "Shape of Design Matrix", np.shape(designMatrix)
    designMatrix_train = np.array(np.reshape(designMatrix, 
        (ds.get_num_examples(), 1, 28, 28) ), dtype=np.float32)
    # 2. find activations
    X = findActivations(model_name, designMatrix_train, which_layer=2, 
                              allFeats=False, bScalar=True, cache=False)
    print 'Done Loading  data'
    if cache:
        np.save(open(filename_X, 'w'), X)
        print 'saved X to', filename_X
        np.save(open(filename_Y, 'w'), Y)
        print 'saved Y to', filename_Y
    
    X_train = X[:test_start_idx]
    Y_train = Y[:test_start_idx]
    X_test = X[test_start_idx:]
    Y_test = Y[test_start_idx:]
        
    return (X_train, Y_train, X_test, Y_test)

    
'''
rf_cl:     random forest classifier as a function
X_test:    design matrix
Y_test:    Does it work for one-hot?
'''
def predictionScores(cl, X_test, Y_test):
    print 'Accuracy Score = ', cl.score(X_test, Y_test)
    #print 'Cross Entropy Loss' #not applicable for random forests

def rfOnActivationsPerformance(model_name, data):
    X_train, Y_train, X_test, Y_test= prepXY(data, model_name)
    for i in range(10):
        print 'RF Trial', i
        trainRF(X_train, Y_train, X_test, Y_test, model_name)
    trainSVM(X_train, Y_train, X_test, Y_test, model_name, one_vs_rest=True)
    trainSVM(X_train, Y_train, X_test, Y_test, model_name, one_vs_rest=False)
    #predictionScores(rf_cl, X_train, Y_train)    0.99
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-layer', action="store", default=2, type=int)
    parser.add_argument('-paramsname', action="store", default='plankton_conv_visualize_model.pkl.params')
    parser.add_argument('-data', action="store", default='')
    allArgs = parser.parse_args()
    layer = allArgs.layer
    model_name = allArgs.paramsname
    rfOnActivationsPerformance(model_name=model_name, data=allArgs.data)