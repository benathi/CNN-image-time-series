'''
This is a wrapper for plankton data
'''
import os, inspect, pickle, random
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

''' 
TODO: 
make it take argument which_classes which specify the class number in the subsets classNames
'''
class PlanktonData(DenseDesignMatrix):
    def __init__(self, which_set='train', rotate='true',
                 which_classes=None, maxPixel='40'):
        ''' Converting arguments to directly usable params '''
        classNames_all = ['diatom_chain_tube', 'fish_larvae_thin_body', 'echinoderm_larva_seastar_brachiolaria'
                      #,'siphonophore_calycophoran_sphaeronectes','trochophore_larvae'
                      ]
        classNames = []
        if which_classes != None:
            for _ind in which_classes:
                classNames.append(classNames_all[_ind])
        else:
            classNames = classNames_all
        maxPixel = int(maxPixel)
        imageSize = maxPixel*maxPixel
        Y_oneHot = None
        # 0. Loading data
        #current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        print '\n\nLoading Plankton Data (Subset) which_set = ', which_set 
        X , Y, Y_info, classDict =  \
        pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain40.p'), 'rb'))   
        print 'Done Loading Data'
        # 1. Set class names
        numClasses = len(classNames)
        self.class_names = classNames
        dictNameToIndex = {}
        for key in classDict:
            dictNameToIndex[classDict[key][0]] = key
        
        # 2. select out only the indices that are in classNames
        class_numbers = [dictNameToIndex[key] for key in classNames]
        relevant_indices = []
        for i, el in enumerate(Y):
            if el in class_numbers:
                relevant_indices.append(i)
        
        # 3. slice X and Y
        X = X[relevant_indices,:imageSize]
        Y = np.array(Y, dtype=np.uint)[relevant_indices]
        X = 1.0 - X
        for i in range(len(Y)):
            _index = class_numbers.index(Y[i])
            #print 'Y[i]= %d. Index = %d'  % (Y[i], _index)
            Y[i] = _index
        
        # 4. Rotating 
        if rotate == 'true':
            X_rotated, Y_rotated = generateRotatedData(X,Y, MAX_PIXEL=maxPixel)
        else:
            X_rotated, Y_rotated = X,Y
        dataSize_afterRotated = X_rotated.shape[0]
        
        # 5. convert to one-hot
        Y_oneHot = np.zeros((np.shape(X_rotated)[0], numClasses))
        for i, val in enumerate(Y_rotated):
            Y_oneHot[i,val] = 1.0
        Y_rotated = Y_oneHot
        
        # 6. slice for train/test after rotating
        if which_set == 'train':
            start = 0
            end  = 100*(int(0.8*dataSize_afterRotated)/100)
        elif which_set == 'valid':
            start = 100*(int(0.8*dataSize_afterRotated)/100)
            end = 100*(dataSize_afterRotated/100)
        X_rotated = X_rotated[start:end]
        Y_rotated = Y_rotated[start:end]
        
        ''' print out information about this data '''
        
        print 'Class Names and Index'
        for i, (name, num) in enumerate(zip(classNames, class_numbers)):
            print 'Index = %d. original class number = %d\t class name= %s. size = %d' % (i, 
                                                                                            num, name,
                                                                                            classDict[num][1])
        print 'Number of classes=\t', numClasses
        print 'train set is the first 80%. validation set is the last 20%'
        print 'Sample Size (train + cv) After Rotating = ', dataSize_afterRotated
        print 'Sample size of returned batch', X_rotated.shape[0]
        
        super(PlanktonData, self).__init__(X=X_rotated, y=Y_rotated)
    
# The pixel values are already from 0 to 1    

def generateRotatedData(X,Y, MAX_PIXEL=40):
    from skimage.transform import rotate
    from skimage.io import imshow
    from matplotlib import pyplot as plt
    
    num_samples = np.shape(X)[0]
    X = np.reshape(X, (num_samples, MAX_PIXEL, MAX_PIXEL))
    num_rotations = 12
    new_num_samples = num_rotations*num_samples
    Xr = np.zeros( (new_num_samples, MAX_PIXEL, MAX_PIXEL) )
    Yr = np.ones( (new_num_samples))
    for rot  in range(num_rotations):
        #print "Rotation", rot
        for i in xrange(num_samples):
            Xr[rot*num_samples + i] = rotate(X[i], angle=360.*rot/float(num_rotations), mode='constant', cval=0.)
        # visualize for sanity check
            if False and i == 0:
                plt.imshow(Xr[num_samples*rot])
                plt.show()
        Yr[rot*num_samples:(rot+1)*num_samples] = Y
    Xr = np.reshape(Xr, (new_num_samples, MAX_PIXEL*MAX_PIXEL))
    
    # shuffle
    np.random.seed(987)
    randPermutation = np.random.permutation(new_num_samples)
    Xr = Xr[randPermutation]
    Yr = Yr[randPermutation]
    print 'Done Rotating Data'
    return Xr, Yr


if __name__=='__main__':
    ds = PlanktonData('train')
