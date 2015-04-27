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
            elif maxPixel == 95:
                X, Y, _, classDict = dumpPlanktonData(maxPixel=95, cache=False)
                 
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
            elif maxPixel == 95:
                X, Y, _, classDict = dumpPlanktonData(maxPixel=95, cache=False)
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
                stop = int(X.shape[0]/100)*100
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
                stop = int(X.shape[0]/100)*100

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
    X = X[:,:maxPixel*maxPixel]
    num_samples = np.shape(X)[0]
    print 'initial shape', X.shape
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

def dumpPlanktonData(maxPixel=40, cache=False):
    import re, glob
    from skimage.io import imread
    from skimage.transform import resize
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

if __name__=='__main__':
    ds = PlanktonData('report')
