import numpy as np


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
def prepXY(model_name, data_spec, which_layer, maxPixel, rawX_test):
    # 1. get data
    #rawX_train, Y_train = getRawData(data_spec, 'train', maxPixel)
    #rawX_cv, Y_cv = getRawData(data_spec, 'valid', maxPixel)
    
    # combine CV to test
    #rawX_train = np.concatenate((rawX_train, rawX_cv), axis=0)
    #Y_train = np.concatenate((Y_train, Y_cv), axis=0)
    X_test = findActivations(model_name, [rawX_test], which_layer, maxPixel)[0]
    # 2. find activations
    print 'Done Finding Activations'
    return X_test

'''
rf_cl:     random forest classifier as a function
X_test:    design matrix
Y_test:    Does it work for one-hot?
'''
def predictionScores(cl, X_test, Y_test):
    print 'Accuracy Score = ', cl.score(X_test, Y_test)

def rfOnActivationsPerformance(model_name, data_spec, num_models, maxPixel):
    print '\n-----#####-----#####-----#####-----#####-----#####-----#####'
    print 'Bagging on CNN'
    print 'pklname', model_name
    print 'data_spec', data_spec
    print 'num_modesl', num_models
    print 'maxPixel', maxPixel
    print 'Obtaining Activations'
    rawX_test, Y_test = getRawData(data_spec, 'test', maxPixel)
    X_test_acc = None
    for i in range(num_models):
        model_name_bag = model_name + '_bag'+ str(i) + '.pkl'
        X_test= prepXY(model_name_bag, data_spec, -1, maxPixel, rawX_test)
        if X_test_acc is None:
            X_test_acc = X_test
        else:
            X_test_acc += X_test
        print '# of models', (i+1)
        Y_predicted = np.argmax(X_test_acc, axis=1)
        print 'prediction score', np.sum(Y_test == Y_predicted)/(1.*len(Y_test))
        print ''

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_models', action="store", default=5, type=int)
    #parser.add_argument('-yaml', action="store", default='plankton_conv_visualize_model.yaml')
    parser.add_argument('-pklname', action="store", default='model_files/plankton_conv_visualize_model_CPU.pkl')
    parser.add_argument('-data', action="store", default='pylearn2_plankton.planktonDataConsolidated')
    parser.add_argument('-maxpix', action="store", default=28, type=int)
    allArgs = parser.parse_args()
    rfOnActivationsPerformance(model_name=allArgs.pklname,
                               data_spec=allArgs.data,
                               num_models=allArgs.num_models,
                               maxPixel=allArgs.maxpix)
