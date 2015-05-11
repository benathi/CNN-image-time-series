'''
Created on Apr 18, 2015

@author: ben
'''
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
import sklearn.feature_selection

def treeImportance(X_train, Y_train, X_test, Y_test, model_name, which_layer, cache=False, n_estimators=400, verbose=False):
    rf_filename = model_name.split('.')[0] + 'layer' + str(which_layer) + '.png'
    print 'CNN model name', model_name
    #print 'Training RF - RF Model Filename =', rf_filename
    forest = RandomForestClassifier(n_estimators=400)
    forest.fit(X_train, Y_train)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    if verbose:
        for f in range(X_train.shape[1]):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    print 'Total number of features', len(indices)
    thres = 1/(100.*len(indices))
    print 'Threshold = ', thres
    print 'Number of features with importance > threshold', np.sum(importances > thres)
    
    
    
    """
    print '\n Feature Importance Using Univariate Selection'
    results = sklearn.feature_selection.univariate_selection.f_regression(X_train, Y_train, center=False)
    F_values, p_values =  results
    for f,p in zip(F_values, p_values):
        print 'f=%e. p=%e' % (f, p)
    """
    # Plot the feature importances of the forest
    '''
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(64), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(64), indices)
    plt.xlim([-1, 64])
    plt.savefig(rf_filename)
    plt.show()
    '''
"""
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
"""

def findActivations(model_name, listX_raw, which_layer, maxPixel):
    # 1. load model file
    import theano
    from pylearn2.utils import serial
    model = serial.load(model_name)
    print 'Model input space is ', model.get_input_space()
    
    # 2. find activations at that layer
    num_x = len(listX_raw)
    activation_list = []
    for X in listX_raw:
        m = X.shape[0]
        X = np.reshape(X, (m,1,maxPixel, maxPixel))
        X = np.swapaxes(X,1,2)
        X = np.swapaxes(X,2,3)
        print 'XReport type = {}. Dimension = {}'.format(type(X), np.shape(X))
        activation = None
        batch_size = 100
        for batchIndex in range(m/batch_size):
            _input = np.array(X[batchIndex*batch_size:(batchIndex+1)*batch_size],
                    dtype=theano.config.floatX)
            fprop_results = model.fprop(theano.shared(_input,
                            name='XReport'), return_all=True)[which_layer].eval()
            if batchIndex==0:
                print 'fprop_results shape', fprop_results.shape
            if activation is None:
                activation = fprop_results
            else:
                activation = np.concatenate((activation, fprop_results), axis=0)
        # need to flatten to be (num points, num features)
        print 'Activation shape before reshape', activation.shape
        activation = np.reshape(activation, (m,-1))
        print 'After reshape', activation.shape
        activation_list.append(activation)
    return activation_list
'''
For example, for model0. layer=2
1. feature 39 (0.025109)
2. feature 32 (0.022700)
3. feature 4 (0.021974)
4. feature 52 (0.021880)
5. feature 56 (0.021431)
6. feature 6 (0.021348)
7. feature 61 (0.020665)
8. feature 12 (0.020435)
9. feature 18 (0.019761)
10. feature 14 (0.019326)
11. feature 60 (0.018864)
12. feature 58 (0.018771)
13. feature 28 (0.018409)
14. feature 19 (0.018289)
15. feature 2 (0.018277)
16. feature 24 (0.018189)
17. feature 59 (0.017989)
18. feature 51 (0.017859)
19. feature 7 (0.017734)
20. feature 44 (0.017532)
21. feature 33 (0.017475)
22. feature 63 (0.017463)
23. feature 37 (0.017441)
24. feature 55 (0.016884)
25. feature 9 (0.016636)
26. feature 47 (0.016486)
27. feature 1 (0.016179)
28. feature 15 (0.015949)
29. feature 8 (0.015888)
30. feature 16 (0.015621)
31. feature 27 (0.015582)
32. feature 0 (0.015243)
33. feature 34 (0.015183)
34. feature 45 (0.015145)
35. feature 5 (0.015095)
36. feature 31 (0.015034)
37. feature 21 (0.014986)
38. feature 40 (0.014964)
39. feature 23 (0.014954)
40. feature 36 (0.014947)
41. feature 57 (0.014930)
42. feature 20 (0.014702)
43. feature 62 (0.014554)
44. feature 10 (0.014413)
45. feature 50 (0.014130)
46. feature 53 (0.014121)
47. feature 38 (0.014023)
48. feature 41 (0.013897)
49. feature 43 (0.013812)
50. feature 22 (0.013710)
51. feature 54 (0.013300)
52. feature 46 (0.013233)
53. feature 3 (0.013207)
54. feature 35 (0.013144)
55. feature 42 (0.013111)
56. feature 13 (0.013042)
57. feature 49 (0.012216)
58. feature 26 (0.011856)
59. feature 25 (0.011087)
60. feature 48 (0.010726)
61. feature 17 (0.009968)
62. feature 30 (0.009122)
63. feature 11 (0.000000)
64. feature 29 (0.000000)
'''
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
    print 'Accuracy Score = ', cl.score(X_test, Y_test)

def rfOnActivationsPerformance(model_name, data_spec, which_layer, maxPixel):
    print '\n-----#####-----#####-----#####-----#####-----#####-----#####'
    print 'Finding Importance of Activations as Features for Random Forests'
    print 'pklname', model_name
    print 'data_spec', data_spec
    print 'which_layer (max layer)', which_layer
    print 'maxPixel', maxPixel
    print 'Obtaining Activations'
    
    for layer in range(which_layer):
        layer = 4
        print '**************************************************************'
        X_train, Y_train, X_test, Y_test= prepXY(model_name, data_spec, layer, maxPixel)
        print 'layer =', layer
        treeImportance(X_train, Y_train, X_test, Y_test, model_name, layer)
        break
    
    
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