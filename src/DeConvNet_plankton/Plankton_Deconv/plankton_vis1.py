'''
Created on Apr 1, 2015

@author: ben

Visualize all 
'''

import os
import sys
import numpy as np
import cPickle
from matplotlib import pyplot as plt


# in order to import customeized classes
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
class_path = parent_folder + '/' + 'utils'
if class_path not in sys.path:
    sys.path.append( class_path )
    
    
from CPRStage import CPRStage_Up,CPRStage_Down
from utils import tile_raster_images
import theano
theano.config.floatX = 'float32'

NUM_C = 1
size_l0 = 32
size_l1 = (size_l0-4)/2 # 14
size_l2 = (size_l1-4)/2 # 5

MAX_PIXEL = 28

model_directory = "../cifar10_sampleCode/Example/"
trainedModelPath = "../../pylearn2_plankton/model_files/"

def activation( a ):
    return ( np.abs(a) + a ) /2 # ReLU max(0,a)

''' This file is run if the pickle file does not exist '''
def loadSamplePlanktons(numSamples=100, rotate=False):
    if not rotate:
        from pylearn2_plankton.planktonDataPylearn2 import PlanktonData
        ds = PlanktonData(which_set='train')
        designMatrix = ds.get_data()[0] # index 1 is the label
        print "Shape of Design Matrix", np.shape(designMatrix)
        designMatrix = np.reshape(designMatrix, 
                                  (ds.get_num_examples(), 1, MAX_PIXEL, MAX_PIXEL) )
        if numSamples != 'All':
            return np.array(designMatrix[:numSamples,...], dtype=np.float32)
        else:
            return np.array(designMatrix, dtype=np.float32)
    else:
        print "Loading Rotated Data"
        designMatrix = np.load(open(os.path.join(os.environ['PYLEARN2_DATA_PATH'] ,'planktonTrainRotatedX.p'), 'r'))
        return np.reshape(np.array(designMatrix[:numSamples,...], dtype=np.float32),
                          (numSamples,1,MAX_PIXEL,MAX_PIXEL))


def example1(model_file_name = "plankton_conv_visualize_model.pkl.params",
             rotatedSample=False,
             bScalar=False):
    """
    In this example, I visulize what the 3rd layer 'see' altogether.
    By set none of feature maps in 3rd layer to zero.
    """

    print "Loading plankton model..."
    model_file = open( trainedModelPath + model_file_name, 'r')
    params = cPickle.load( model_file )
    model_file.close()

    layer0_w = params[-2]
    layer0_b = params[-1]
    layer1_w = params[-4]
    layer1_b = params[-3]
    layer2_w = params[-6]
    layer2_b = params[-5]
    
    print 'layer0_w shape', np.shape(layer0_w)
    print 'layer0_b shape', np.shape(layer0_b)
    print 'layer1_w shape', np.shape(layer1_w)
    print 'layer1_b shape', np.shape(layer1_b)
    print 'layer2_w shape', np.shape(layer2_w)
    print 'layer2_b shape', np.shape(layer2_b)
    
    '''
    Note: after applying convolution, our weight is a weight per pixel
    whereas the weight here is per the whole array. Fix this by averaging for now
    '''
    if bScalar:
        print "Using B as scalar"
        layer0_b = np.mean(layer0_b, axis=(1,2))
        layer1_b = np.mean(layer1_b, axis=(1,2))
        layer2_b = np.mean(layer2_b, axis=(1,2))
    else:
        print "Using B as matrix. B has the following shapes"
        print "Layer 0", np.shape(layer0_b)
        print "Layer 1", np.shape(layer1_b)
        print "Layer 2", np.shape(layer2_b)
    
    # forward
    # filter shape is shape of layer0_w
    up_layer0 = CPRStage_Up( image_shape = (1,NUM_C,28,28), filter_shape = (32,NUM_C,5,5),
                            poolsize = 2 , W = layer0_w, b = layer0_b, 
                            activation = activation)
                            
    up_layer1 = CPRStage_Up( image_shape = (1,32,12,12), filter_shape = (48,32,5,5), 
                            poolsize = 2,W = layer1_w, b = layer1_b ,
                            activation = activation)
                            
    up_layer2 = CPRStage_Up( image_shape = (1,48,4,4), filter_shape = (64,48,3,3), 
                            poolsize = 2,W = layer2_w, b = layer2_b ,
                            activation = activation)
    # backward
    down_layer2 = CPRStage_Down( image_shape = (1,64,2,2), filter_shape = (48,64,3,3), 
                                poolsize = 2,W =layer2_w, b = layer2_b,
                                activation = activation)
                                
    down_layer1 = CPRStage_Down( image_shape = (1,48,4*2,4*2), filter_shape = (32,48,5,5), 
                                poolsize = 2,W =layer1_w, b = layer1_b,
                                activation = activation)
                                
    down_layer0 = CPRStage_Down( image_shape = (1,32,28-4,28-4), filter_shape = (NUM_C,32,5,5), 
                                poolsize = 2,W = layer0_w, b = layer0_b,
                                activation = activation)

    
    # load sample images
    print 'Loading sample images...'
    #f = open( model_directory + 'SubSet25.pkl', 'r' )
    #input = cPickle.load( f )
    #f.close()
    input = loadSamplePlanktons(rotate=rotatedSample)

    print 'Sample Images Shape', np.shape(input)

    output = np.ndarray( input.shape )
    num_of_sam = input.shape[0]
    print 'Total %d images' % num_of_sam
    
    for i in xrange(num_of_sam):
        print '\tdealing with image %d' % (i+1)
        l0u , sw0 = up_layer0.GetOutput( input[i].reshape(1,NUM_C,MAX_PIXEL,MAX_PIXEL) )
        #if i == 0:  print "After l0u"
        l1u  , sw1 = up_layer1.GetOutput( l0u )
        #if i == 0:  print "After l1u"
        l2u  , sw2 = up_layer2.GetOutput( l1u )
        if i == 0:
            print "Shape of l0u", np.shape(l0u)
            print "Shape of l1u", np.shape(l1u)
            print "Shape of l2u", np.shape(l2u)
        l2d = down_layer2.GetOutput( l2u, sw2 )
        #if i == 0:  print "After l2d"
        l1d = down_layer1.GetOutput( l2d, sw1 )
        #if i == 0:  print "After l1d"
        l0d = down_layer0.GetOutput( l1d , sw0 )
        #if i == 0:  print "After l0d"
        output[i] = l0d
        
    # from bc01 to cb01
    input = np.transpose( input, [ 1, 0, 2, 3 ])         
    output = np.transpose( output, [ 1, 0, 2, 3 ])
    
    # flatten
    input = input.reshape( [ NUM_C, num_of_sam, MAX_PIXEL*MAX_PIXEL ])
    output = output.reshape( [ NUM_C, num_of_sam, MAX_PIXEL*MAX_PIXEL ])
    print "Shape of Input",  np.shape(input)
    # transform to fit tile_raster_images    
    input = tuple( [ input[0] for i in xrange(3)] + [None] )    # quick hack for gray scale 
    output = tuple( [ output[0] for i in xrange(3)] + [None] )  # quick hack for gray scale
    print "Len of Input = ", len(input)
    for el in input:
        print np.shape(el)
    input_map = tile_raster_images( input, img_shape = (MAX_PIXEL,MAX_PIXEL), tile_shape = (10,10), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
                                    
    output_map = tile_raster_images( output, img_shape = (MAX_PIXEL,MAX_PIXEL), tile_shape = (10,10), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
    
    bigmap = np.append( input_map, output_map, axis = 1 )      

    plt.imshow(bigmap)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rotate', action="store_true", default=False)
    parser.add_argument('-paramsname', action="store", default='plankton_conv_visualize_model.pkl.params')
    parser.add_argument('-bScalar', action="store_true", default=False)
    results = parser.parse_args()
    rotate = results.rotate
    model_name = results.paramsname
    bScalar = results.bScalar
    print 'Loading Params from', model_name
    example1(model_name, rotate, bScalar)
    