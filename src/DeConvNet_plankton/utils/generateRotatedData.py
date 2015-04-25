'''
Created on Apr 2, 2015

@author: ben
'''
import os, pickle
import numpy as np
from skimage.transform import rotate
from skimage.io import imshow
from matplotlib import pyplot as plt
MAX_PIXEL = 28
imageSize = MAX_PIXEL*MAX_PIXEL

def dumpTrainDataRotated(which_set='train_rotate'):
    if True:
        if which_set == 'train_rotate':
            # 0. Loading data
            #current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            X, Y, _, _ = \
            pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain.p'), 'rb'))   
            X = np.array(1.0 - X[:,:imageSize], dtype=np.float32)
            Xr, Yr = generateRotatedData(X,Y)
            print "Finished Rotating"
            #pickle.dump( (Xr,Yr) , open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrainRotated.p'), 'wb'))
            np.save(open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrainRotatedX.p'), 'wb'), Xr)
            np.save(open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrainRotatedY.p'), 'wb'), Yr)
            print "Done dumping to file"
            
'''
Returns a tuple
'''
def generateRotatedData(X,Y, MAX_PIXEL=28, flatten=True):
    num_samples = np.shape(X)[0]
    X = np.reshape(X, (num_samples, MAX_PIXEL, MAX_PIXEL))
    num_rotations = 12
    new_num_samples = num_rotations*num_samples
    Xr = np.zeros( (new_num_samples, MAX_PIXEL, MAX_PIXEL) , dtype=np.float32)
    Yr = np.ones( (new_num_samples) )
    for rot  in range(num_rotations):
        print "Rotation", rot
        for i in xrange(num_samples):
            Xr[rot*num_samples + i] = rotate(X[i], angle=360.*rot/float(num_rotations), mode='constant', cval=0.)
        # visualize for sanity check
            if False and i == 0:
                plt.imshow(Xr[num_samples*rot])
                plt.show()
        Yr[rot*num_samples:(rot+1)*num_samples] = Y
    if flatten:
        Xr = np.reshape(Xr, (new_num_samples, MAX_PIXEL*MAX_PIXEL))
    
    # shuffle
    randPermutation = np.random.permutation(new_num_samples)
    Xr = Xr[randPermutation]
    Yr = Yr[randPermutation]
    return Xr, Yr
'''
input: of shape (b, c, 0, 1)
output: of shape (b, c, 0, 1)
'''
def generateRotatedDataPreserveDim(X,Y, num_rotations=12):
    num_samples = np.shape(X)[0]
    num_chans = np.shape(X)[1]
    max_pixel = np.shape(X)[2]  # assume square
    new_num_samples = num_rotations*num_samples
    Xr = np.zeros( (new_num_samples, num_chans,  max_pixel, max_pixel) , dtype=np.float32)
    Yr = np.ones( (new_num_samples) )
    for rot  in range(num_rotations):
        print "Rotation", rot
        for i in xrange(num_samples):
            for ch in range(num_chans):
                Xr[rot*num_samples + i, ch] = rotate(X[i,ch], angle=360.*rot/float(num_rotations), mode='constant', cval=0.)
                # visualize for sanity check
                if False and i == 0:
                    plt.imshow(Xr[num_samples*rot,ch])
                    plt.show()
        Yr[rot*num_samples:(rot+1)*num_samples] = Y
    # shuffle
    np.random.seed(14249)
    randPermutation = np.random.permutation(new_num_samples)
    Xr = Xr[randPermutation]
    Yr = Yr[randPermutation]
    return Xr, Yr
    
if __name__ == "__main__":
    dumpTrainDataRotated()