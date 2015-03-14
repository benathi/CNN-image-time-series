'''
This is a wrapper for plankton data
'''
import os, inspect, pickle
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

class PlanktonData(DenseDesignMatrix):
    def __init__(self, which_set='train', shuffle=False,
                 start=None, stop=None,
                 preprocessor=None):
        maxPixel = 28
        imageSize = maxPixel*maxPixel
        Y_oneHot = None
        if which_set == 'train':
            # 0. Loading data
            #current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
            X , Y, _, classDict =  \
            pickle.load( open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrain.p'), 'rb'))   
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
            ''' Next: report data set '''
        elif which_set == 'report':
            print 'Loading Report Data'
            from skimage.io import imread
            from skimage.transform import resize
            import re, glob
            print 'Loading Plankton Testing Data'
            dataPath = '../../data/plankton/test'
            m = 0
            MAX = 600 # delete MAX
            for filename in glob.glob(os.path.join(dataPath, '*.jpg'))[:MAX]:
                if filename[-4:] != ".jpg":
                    continue
                m += 1
            
            print 'Number of Inputs = %d' % (m)
            XReport = np.empty((m, imageSize))
            Y_infoName = []
            i = -1
            for filename in glob.glob(os.path.join(dataPath, '*.jpg'))[:MAX]:
                if filename[-4:] != ".jpg":
                    continue
                i += 1
                image = imread(filename, as_grey=True)
                image = resize(image, (maxPixel, maxPixel))
                XReport[i,:] = np.reshape(image, (1, imageSize))
                shortFileName = re.match(r'.*/(.*)', filename).group(1)
                #print shortFileName
                Y_infoName.append( (shortFileName) )
                
            XReport = 1.0 - XReport
            Y_infoName = np.array(Y_infoName)
            self.Y_infoName = Y_infoName
            X = XReport
        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            Y_oneHot = Y_oneHot[start:stop, :]
            assert X.shape[0] == Y_oneHot.shape[0]
        print 'Done Loading Data - PlanktonData'
        super(PlanktonData, self).__init__(X=X, y=Y_oneHot)
    
# The pixel values are already from 0 to 1    

if __name__=='__main__':
    ds = PlanktonData('report')