'''
Created on Apr 1, 2015

@author: ben
'''

#-*- coding: utf-8 -*- 
import os
import sys
import numpy as np
import cPickle
import theano
import theano.tensor as T
from matplotlib import pyplot as plt
from heapq import *

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
from Layers import ConvPoolLayer,relu_nonlinear
theano.config.floatX = 'float32'

model_directory = "../cifar10_sampleCode/Example/"
trainedModelPath = "../../pylearn2_plankton/model_files/"
NUM_C = 1
MAX_PIXEL = 28


def activation( a ):
    return ( np.abs(a) + a ) /2


class Pairs( object ):
    """
    If we store  ( activation,sample ) using tuple/list, when two activation
    value equal, the heappushpop algorithm will compare sample to sort the 
    list/tuple and than raise an 'any() or all()' error. To avoid this, we 
    design an class which implement the __lt__ method to ensure the heap
    algorithm always compare to activation.
    """
    def __init__( self, activation, input ):
        self.act = activation
        self.sam = input
        
    def __lt__( self, obj ):
        """
        Customized __lt__ to avoid compare input
        """
        # compare Pairs object by act value
        return self.act < obj.act

class DeConvNet( object ):
    
    def __init__( self, model_name="plankton_conv_visualize_model.pkl.params"):

        print "Loading plankton model..."
        model_file = open( trainedModelPath + model_name, 'r')
        params = cPickle.load( model_file )
        model_file.close()
        
        layer0_w = params[-2]
        layer0_b = params[-1]
        layer1_w = params[-4]
        layer1_b = params[-3]
        layer2_w = params[-6]
        layer2_b = params[-5]
        
        '''
        Note: after applying convolution, our weight is a weight per pixel
        whereas the weight here is per the whole array. Fix this by averaging for now
        '''
        layer0_b = np.mean(layer0_b, axis=(1,2))
        layer1_b = np.mean(layer1_b, axis=(1,2))
        layer2_b = np.mean(layer2_b, axis=(1,2))

        # compile theano function for efficient forward propagation
        x = T.tensor4('x')
 
        layer0 = ConvPoolLayer( input = x, image_shape = (1,NUM_C,28,28), 
                                filter_shape = (32,NUM_C,5,5), W = layer0_w,
                                b = layer0_b, poolsize=(2, 2), 
                                activation = relu_nonlinear)
                                
        layer1 = ConvPoolLayer( input = layer0.output, image_shape = (1,32,12,12), 
                                filter_shape = (48,32,5,5), W = layer1_w, 
                                b = layer1_b, poolsize=(2, 2), 
                                activation = relu_nonlinear)
        
        layer2 = ConvPoolLayer( input = layer1.output, image_shape = (1,48,4,4), 
                                filter_shape = (64,48,3,3), W = layer2_w, 
                                b = layer2_b, poolsize=(2, 2), 
                                activation = relu_nonlinear) 
        print "Compiling theano.function..."
        self.forward = theano.function( [x], layer2.output )
        self.forward2 = theano.function( [x], layer2.output)
        self.forward1 = theano.function( [x], layer1.output)
        self.forward0 = theano.function( [x], layer0.output)
                               
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

        self.Stages = [ up_layer0, up_layer1, up_layer2,
                           down_layer2, down_layer1, down_layer0]
        
    def DeConv( self, input, kernel_index, which_layer=2 ):

        assert kernel_index != None        
        #print "Initial Input Shape", np.shape(input)
        l0u , sw0 = self.Stages[0].GetOutput( input.reshape(1,NUM_C,MAX_PIXEL,MAX_PIXEL) )
        l1u  , sw1 = self.Stages[1].GetOutput( l0u )
        l2u  , sw2 = self.Stages[2].GetOutput( l1u )
        
        # only visualize selected kernel - set other output map to zeros
        if which_layer == 2:
            l2u[0,:kernel_index,...]*=0
            l2u[0,kernel_index+1:,...]*=0
            
            l2d = self.Stages[3].GetOutput( l2u, sw2 )
            l1d = self.Stages[4].GetOutput( l2d, sw1 )
            l0d = self.Stages[5].GetOutput( l1d , sw0 )
        elif which_layer == 1:
            l1u[0,:kernel_index,...]*=0
            l1u[0,kernel_index+1:,...]*=0
            
            #l2d = self.Stages[3].GetOutput( l2u, sw2 )
            l1d = self.Stages[4].GetOutput( l1u, sw1 )
            l0d = self.Stages[5].GetOutput( l1d , sw0 )
        elif which_layer == 0:
            l0u[0,:kernel_index,...]*=0
            l0u[0,kernel_index+1:,...]*=0
            
            #l2d = self.Stages[3].GetOutput( l2u, sw2 )
            #l1d = self.Stages[4].GetOutput( l2d, sw1 )
            l0d = self.Stages[5].GetOutput( l0u , sw0 )
        
        #l2d = self.Stages[3].GetOutput( l2u, sw2 )
        #l1d = self.Stages[4].GetOutput( l2d, sw1 )
        #l0d = self.Stages[5].GetOutput( l1d , sw0 )
  
        return l0d
        
    
def findmaxactivation( Net, samples, num_of_maximum, kernel_list, which_layer=2):
                           
    """
    Return a list of heaps. Each heap(list) corresponding to specific kernel 
    marked by kernel_list. In each heap are num_of_maximum Pairs whose samples
    yield max activation to corresponding kernel in given samples.

    :type Net: Deconvnet which the last layer has feature maps heights and 
                weights only 1.( manually cut off edge of input )
    :param Net: class Net with method Forward and Deconv( interface )

    :type samples: 4D-array int the form of BC01
    :param samples: numpy.ndarray

    :type num_of_maximum: number of maximums to be restored in each heap
    :param num_of_maximum: int

    :type kernel_list: the index of kernels to be visulized( start from 0 )
    :param kernel_list: list of int
    """ 
    print "Function: Find Max Activation"
    print "num_of_maximum = ", num_of_maximum
    Heaps = { kernel_i: [ Pairs( -100, -i) for i in xrange(num_of_maximum)]\
                    for kernel_i in  kernel_list  }
    #print "Printing Initial Heaps"
    #printHeaps(Heaps)
    
    index = 0
    print "Total %d samples" % samples.shape[0]   
    print "Type of samp", type(samples[0])
    print "Type of element", type(samples[0,0,0,0])
    for sam in samples:
        index += 1
        if index % 100 == 0:
            print "pushpoping ",index,"th sample"  
        # from 3-dim to 4-dim
        sam = sam.reshape((1,)+sam.shape )      
        if which_layer == 2:
            activate_value = Net.forward2(sam)
        elif which_layer == 1:
            activate_value = Net.forward1(sam)
        elif which_layer == 0:
            activate_value = Net.forward0(sam)
        # activate_value for all the kernels
        # question: What layer is this?
        if index == 1:
            print "Shape of activate_value = ", np.shape(activate_value) 
            #64 for this example
            #print "Type of Net.forward", type(Net.forward)
            #<class 'theano.compile.function_module.Function'>
        numKernels = np.shape(activate_value)[1]
        newActivateValues = np.zeros((numKernels))
        for i in range(numKernels):
            # using the norm to determine the activate value
            newActivateValues[i] = np.linalg.norm(activate_value[0,i]) 
            #newActivateValues[i] = np.std( activate_value[0,i].flatten())
             
        for kernel_i in kernel_list:
            heappushpop( Heaps[kernel_i], Pairs( newActivateValues[kernel_i], sam ))
            # heappushpop is from python's heapq
            # maximum heap
    printHeaps(Heaps)        
    ####
    return Heaps

def printHeaps(Heaps):
    for key in Heaps:
        print "Kernel {}".format(key)
        for p in Heaps[key]:
            print "\tActivation = {} Input Sam {}".format(p.act, np.shape(p.sam))

def Find_plankton(model_name="plankton_conv_visualize_model.pkl.params", rotate=False):
    """
    Find plankton that activates the given layers most
    """
    which_layer = 2
    
    import plankton_vis1
    samples = plankton_vis1.loadSamplePlanktons(numSamples=3000,rotate=rotate)
    print 'Dimension of Samples', np.shape(samples)
    Net = DeConvNet(model_name)
    
    #kernel_list = [ 2,23,60,12,45,9 ]
    kernel_list = range(0,16)
    
    
    num_of_maximum = 9
    Heaps = findmaxactivation( Net, samples, num_of_maximum, kernel_list, which_layer=which_layer)
    bigbigmap = None # what is this?
    for kernel_index in Heaps:        
        print 'dealing with',kernel_index,'th kernel'
        heap = Heaps[kernel_index]
        this_sams = []
        this_Deconv = []
        for pairs in heap:
            this_sam = pairs.sam
            this_sams.append( this_sam.reshape([NUM_C,MAX_PIXEL,MAX_PIXEL]) )
            this_Deconv.append( Net.DeConv( this_sam, kernel_index, which_layer=which_layer ).reshape([NUM_C,MAX_PIXEL,MAX_PIXEL]) )
        
        this_sams = np.array( this_sams )
        this_sams = np.transpose( this_sams, [ 1, 0, 2, 3 ])
        this_sams = this_sams.reshape( [ NUM_C, 9, MAX_PIXEL*MAX_PIXEL ])
        this_sams = tuple( [ this_sams[0] for i in xrange(3)] + [None] )    
        
        this_Deconv = np.array( this_Deconv )
        this_Deconv = np.transpose( this_Deconv, [ 1, 0, 2, 3 ])
        this_Deconv = this_Deconv.reshape( [ NUM_C, num_of_maximum, MAX_PIXEL*MAX_PIXEL ])
        this_Deconv = tuple( [ this_Deconv[0] for i in xrange(3)] + [None] )

        this_map = tile_raster_images( this_sams, img_shape = (MAX_PIXEL,MAX_PIXEL), tile_shape = (3,3), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
        this_Deconv = tile_raster_images( this_Deconv, img_shape = (MAX_PIXEL,MAX_PIXEL), tile_shape = (3,3), 
                                   tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                    output_pixel_vals=True)
        this_pairmap = np.append( this_map, this_Deconv, axis = 0)

        if bigbigmap == None:
            bigbigmap = this_pairmap
            segment_line = 255*np.ones([bigbigmap.shape[0],1,4],dtype='uint8')
        else:
            bigbigmap = np.append(bigbigmap, segment_line, axis = 1)            
            bigbigmap = np.append(bigbigmap, this_pairmap, axis = 1)
            
            
    plt.imshow(bigbigmap)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rotate', action="store_true", default=False)
    parser.add_argument('-paramsname', action="store", default='plankton_conv_visualize_model.pkl.params')
    results = parser.parse_args()
    rotate = results.rotate
    model_name = results.paramsname
    print 'Loading Params from', model_name
    Find_plankton(model_name, rotate)
    