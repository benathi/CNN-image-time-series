import os
import sys
import numpy as np
import cPickle
from matplotlib import pyplot as plt


# in order to import customeized classes
parent_folder = os.path.abspath('../..')
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

NUM_C = 3
size_l0 = 32
size_l1 = (size_l0-4)/2 # 14
size_l2 = (size_l1-4)/2 # 5

def activation( a ):
    return ( np.abs(a) + a ) /2 # ReLU max(0,a)

def example1():
    """
    In this example, I visulize what the 3rd layer 'see' altogether.
    By set none of feature maps in 3rd layer to zero.
    """

    print "Loading model..."
    model_file = open( 'params.pkl', 'r')
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
    print 'layer0_b[0] shape', np.shape(layer0_b[0])
    print 'layer0_b[0] type', type(layer0_b[0]) # just a float
    print 'layer1_w shape', np.shape(layer1_w)
    print 'layer1_b shape', np.shape(layer1_b)
    print 'layer2_w shape', np.shape(layer2_w)
    print 'layer2_b shape', np.shape(layer2_b)

    # forward
    # filter shape is shape of layer0_w
    up_layer0 = CPRStage_Up( image_shape = (1,NUM_C,32,32), filter_shape = (32,NUM_C,5,5),
                            poolsize = 2 , W = layer0_w, b = layer0_b, 
                            activation = activation)
                            
    up_layer1 = CPRStage_Up( image_shape = (1,32,14,14), filter_shape = (50,32,5,5), 
                            poolsize = 2,W = layer1_w, b = layer1_b ,
                            activation = activation)
                            
    up_layer2 = CPRStage_Up( image_shape = (1,50,5,5), filter_shape = (64,50,5,5), 
                            poolsize = 1,W = layer2_w, b = layer2_b ,
                            activation = activation)
    # backward
    down_layer2 = CPRStage_Down( image_shape = (1,64,1,1), filter_shape = (50,64,5,5), 
                                poolsize = 1,W =layer2_w, b = layer2_b,
                                activation = activation)
                                
    down_layer1 = CPRStage_Down( image_shape = (1,50,10,10), filter_shape = (32,50,5,5), 
                                poolsize = 2,W =layer1_w, b = layer1_b,
                                activation = activation)
                                
    down_layer0 = CPRStage_Down( image_shape = (1,32,28,28), filter_shape = (NUM_C,32,5,5), 
                                poolsize = 2,W = layer0_w, b = layer0_b,
                                activation = activation)


    # load sample images
    print 'Loading sample images...'
    f = open( 'SubSet25.pkl', 'r' )
    input = cPickle.load( f )
    f.close()

    print 'Sample Images Shape', np.shape(input)

    output = np.ndarray( input.shape )
    num_of_sam = input.shape[0]
    print 'Totally %d images' % num_of_sam

    for i in xrange(num_of_sam):
        print '\tdealing with %d image...' % (i+1)
        l0u , sw0 = up_layer0.GetOutput( input[i].reshape(1,NUM_C,32,32) )
        l1u  , sw1 = up_layer1.GetOutput( l0u )
        l2u  , sw2 = up_layer2.GetOutput( l1u )
        if i == 0:
            print "Shape of l0u", np.shape(l0u)
            print "Shape of l1u", np.shape(l1u)
            print "Shape of l2u", np.shape(l2u)
        l2d = down_layer2.GetOutput( l2u, sw2 )
        l1d = down_layer1.GetOutput( l2d, sw1 )
        l0d = down_layer0.GetOutput( l1d , sw0 )
        output[i] = l0d
        
    # from bc01 to cb01
    input = np.transpose( input, [ 1, 0, 2, 3 ])         
    output = np.transpose( output, [ 1, 0, 2, 3 ])
    
    # flatten
    input = input.reshape( [ NUM_C, 25, 32*32 ])
    output = output.reshape( [ NUM_C, 25, 32*32 ])

    # transform to fit tile_raster_images    
    input = tuple( [ input[i] for i in xrange(NUM_C)] + [None] )    
    output = tuple( [ output[i] for i in xrange(NUM_C)] + [None] )   
    
    input_map = tile_raster_images( input, img_shape = (32,32), tile_shape = (5,5), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
                                    
    output_map = tile_raster_images( output, img_shape = (32,32), tile_shape = (5,5), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
    
    bigmap = np.append( input_map, output_map, axis = 1 )      

    plt.imshow(bigmap)
    plt.show()

if __name__ == "__main__":
    example1()
