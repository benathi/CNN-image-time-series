'''
This is a wrapper for plankton data
'''
import os, inspect, pickle
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

'''
This consolidated data module supports many pixel sizes and
many types of data set (train, test, valid)
and an option for augmented data
'''
class PlanktonData(DenseDesignMatrix):
    def __init__(self, which_set, maxPixel, rotate=False):
        # to support 28,40,95
        assert maxPixel in [28, 40], \
        'max pixel %r is not supported (only %r supported)' % (maxPixel, [28, 40])
        assert which_set in  ['train', 'valid', 'test'], \
        'which_set %r is not in the list %r' % (which_set, ['train', 'valid', 'test'])
        imageSize = maxPixel*maxPixel
        Y_oneHot = None
        
        print 'Module PlanktonData. which_set=%r\t rotated=%r imagesize=%r' % (which_set, rotate, maxPixel)
        if not rotate:
            ''' All of the non-rotated data '''
            # 0. Loading data
            #current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            if maxPixel == 28:
                X , Y, _, classDict =  \
                pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain.p'), 'rb'))
            elif maxPixel == 40:
                X , Y, _, classDict =  \
                pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain40.p'), 'rb'))  
            # 1. Set class names
            numClasses = 121
            classNames = []
            for i in range(numClasses):
                classNames.append(classDict[i][0])
            self.class_names = classNames
            
            X = X[:,:imageSize]
            X = 1.0 - X
            Y_oneHot = np.zeros((np.shape(X)[0], numClasses))
            for i, val in enumerate(Y):
                Y_oneHot[i,val] = 1.0
        elif rotate:
            ''' all of the rotated data '''
            # 0. Loading data
            #current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            if maxPixel == 28:
                X , Y, _, classDict =  \
                pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain.p'), 'rb'))
            elif maxPixel == 40:
                X , Y, _, classDict =  \
                pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain40.p'), 'rb'))
            # 1. Set class names
            numClasses = 121
            classNames = []
            for i in range(numClasses):
                classNames.append(classDict[i][0])
            self.class_names = classNames
            #file_path = os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrainRotated.p')
            #X,Y = pickle.load(open(file_path, 'rb'))
            X,Y = generateRotatedData(X, Y, maxPixel, num_rotations=12)
            Y_oneHot = np.zeros((np.shape(X)[0], numClasses))
            for i, val in enumerate(Y):
                Y_oneHot[i,val] = 1.0
            ''' Next: report data set '''

        if not rotate:
            if which_set == 'train':
                start = 0
                stop = 25000
            elif which_set == 'valid':
                start = 25000
                stop = 26500
            elif which_set == 'test':
                start = 26500
                stop = X.shape[0]
        else:
            # rotate
            if which_set == 'train':
                start = 0*12
                stop = 25000*12
            elif which_set == 'valid':
                start = 25000*12
                stop = 26000*12
            elif which_set == 'test':
                start = 26000*12
                stop = X.shape[0]

        X = X[start:stop, :]
        Y_oneHot = Y_oneHot[start:stop, :]
        assert X.shape[0] == Y_oneHot.shape[0]
        print 'Done Loading Data - PlanktonData'
        print 'Data Size', (stop-start)
        super(PlanktonData, self).__init__(X=X, y=Y_oneHot)

''' 
    input [numpoints, max_pixel*max_pixel]
    output [numpoints*num_rotations, max_pixel*max_pixel]
'''
def generateRotatedData(X,Y, maxPixel=28, num_rotations=12):
    from skimage.transform import rotate
    num_samples = np.shape(X)[0]
    X = np.reshape(X, (num_samples, maxPixel, maxPixel))
    new_num_samples = num_rotations*num_samples
    Xr = np.zeros( (new_num_samples, maxPixel, maxPixel) , dtype=np.float32)
    Yr = np.ones( (new_num_samples) )
    for rot  in range(num_rotations):
        print "Rotation", rot
        for i in xrange(num_samples):
            Xr[rot*num_samples + i] = rotate(X[i], angle=360.*rot/float(num_rotations), mode='constant', cval=0.)
        Yr[rot*num_samples:(rot+1)*num_samples] = Y
    Xr = np.reshape(Xr, (new_num_samples, maxPixel*maxPixel))
    # shuffle
    np.random.seed(14249)
    randPermutation = np.random.permutation(new_num_samples)
    Xr = Xr[randPermutation]
    Yr = Yr[randPermutation]
    return Xr, Yr


if __name__=='__main__':
    ds = PlanktonData('report')
