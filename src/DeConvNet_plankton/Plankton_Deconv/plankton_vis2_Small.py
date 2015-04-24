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
MAX_PIXEL = 40


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
    
    def __init__( self, model_name="plankton_conv_visualize_model.pkl.params",
                  bScalar=False):

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
        layer3_w = params[-8]
        layer3_b = params[-7]
        layer4_w = params[-10]
        layer4_b = params[-9]     
        
        print 'layer0_w shape', np.shape(layer0_w)
        print 'layer0_b shape', np.shape(layer0_b)
        print 'layer1_w shape', np.shape(layer1_w)
        print 'layer1_b shape', np.shape(layer1_b)
        print 'layer2_w shape', np.shape(layer2_w)
        print 'layer2_b shape', np.shape(layer2_b)
        print 'layer3_w shape', np.shape(layer3_w)
        print 'layer3_b shape', np.shape(layer3_b)
        print 'layer4_w shape', np.shape(layer4_w)
        print 'layer4_b shape', np.shape(layer4_b)
               
        '''
        Note: after applying convolution, our weight is a weight per pixel
        whereas the weight here is per the whole array. Fix this by averaging for now
        '''
        if bScalar:
            print "Using b as scalar"
            layer0_b = np.mean(layer0_b, axis=(1,2))
            layer1_b = np.mean(layer1_b, axis=(1,2))
            layer2_b = np.mean(layer2_b, axis=(1,2))
            layer3_b = np.mean(layer3_b, axis=(1,2))
            layer4_b = np.mean(layer4_b, axis=(1,2))            
        else:
            print "Using b as matrix"

        # compile theano function for efficient forward propagation
        x = T.tensor4('x')
        numLayers = 5 # counting 0
        conv = [5,5,3,3,3]
        pool = [1,2,1,2,2]
        chs = [16, 16, 32, 32, 32]
        print 'conv', conv
        print 'pool', pool
        print 'chs', chs
        img_size = 40
        ap = [0 for i in range(numLayers)]
        ac = [0 for i in range(numLayers)]
        for i in range(numLayers):
            if i == 0:
                ac[i] = img_size - conv[i] + 1
            else:
                print ac[i]
                print ap[i-1]
                print conv[i]
                ac[i] = ap[i-1] - conv[i] + 1
            if ac[i] % pool[i] != 0:
                print 'Warning! Pooling not valid!'
            ap[i] = ac[i]/pool[i]
        print 'img_after conv', ac
        print 'img_after pool', ap
        
        layer0 = ConvPoolLayer( input = x, image_shape = (1,NUM_C,img_size,img_size), 
                                filter_shape = (chs[0],NUM_C,conv[0],conv[0]), W = layer0_w,
                                b = layer0_b, poolsize=(pool[0], pool[0]), 
                                activation = relu_nonlinear)
                                
        layer1 = ConvPoolLayer( input = layer0.output, image_shape = (1,chs[0],ap[0],ap[0]), 
                                filter_shape = (chs[1],chs[0],conv[1],conv[1]), W = layer1_w, 
                                b = layer1_b, poolsize=(pool[1], pool[1]), 
                                activation = relu_nonlinear)
        
        layer2 = ConvPoolLayer( input = layer1.output, image_shape = (1,chs[1],ap[1],ap[1]), 
                                filter_shape = (chs[2],chs[1],conv[2],conv[2]), W = layer2_w, 
                                b = layer2_b, poolsize=(pool[2], pool[2]), 
                                activation = relu_nonlinear) 
        layer3 = ConvPoolLayer( input = layer2.output, image_shape = (1,chs[2],ap[2],ap[2]), 
                                filter_shape = (chs[3],chs[2],conv[3],conv[3]), W = layer3_w, 
                                b = layer3_b, poolsize=(pool[3], pool[3]), 
                                activation = relu_nonlinear) 
        layer4 = ConvPoolLayer( input = layer3.output, image_shape = (1,chs[3],ap[3],ap[3]), 
                                filter_shape = (chs[4],chs[3],conv[4],conv[4]), W = layer4_w, 
                                b = layer4_b, poolsize=(pool[4], pool[4]), 
                                activation = relu_nonlinear)         
        
        print "Compiling theano.function..."
        self.forward = theano.function( [x], layer4.output )
        self.forward4 = theano.function( [x], layer4.output)
        self.forward3 = theano.function( [x], layer3.output)
        self.forward2 = theano.function( [x], layer2.output)
        self.forward1 = theano.function( [x], layer1.output)
        self.forward0 = theano.function( [x], layer0.output)
                               
        up_layer0 = CPRStage_Up( image_shape = (1,NUM_C,img_size,img_size), filter_shape = (chs[0],NUM_C,conv[0],conv[0]),
                            poolsize = pool[0] , W = layer0_w, b = layer0_b, 
                            activation = activation)
                            
        up_layer1 = CPRStage_Up( image_shape = (1,chs[0],ap[0],ap[0]), filter_shape = (chs[1],chs[0],conv[1],conv[1]), 
                            poolsize = pool[1],W = layer1_w, b = layer1_b ,
                            activation = activation)
                            
        up_layer2 = CPRStage_Up( image_shape = (1,chs[1],ap[1],ap[1]), filter_shape = (chs[2],chs[1],conv[2],conv[2]), 
                            poolsize = pool[2],W = layer2_w, b = layer2_b ,
                            activation = activation)
        up_layer3 = CPRStage_Up( image_shape = (1,chs[2],ap[2],ap[2]), filter_shape = (chs[3],chs[2],conv[3],conv[3]), 
                            poolsize = pool[3],W = layer3_w, b = layer3_b ,
                            activation = activation)
        up_layer4 = CPRStage_Up( image_shape = (1,chs[3],ap[3],ap[3]), filter_shape = (chs[4],chs[3],conv[4],conv[4]), 
                            poolsize = pool[4],W = layer4_w, b = layer4_b ,
                            activation = activation)        
        
        # backward
        down_layer4 = CPRStage_Down( image_shape = (1,chs[4],ac[4],ac[4]), filter_shape = (chs[3],chs[4],conv[4],conv[4]), 
                                poolsize = pool[4],W =layer4_w, b = layer4_b,
                                activation = activation)
        
        down_layer3 = CPRStage_Down( image_shape = (1,chs[3],ac[3],ac[3]), filter_shape = (chs[2],chs[3],conv[3],conv[3]), 
                                poolsize = pool[3],W =layer3_w, b = layer3_b,
                                activation = activation)
        
        down_layer2 = CPRStage_Down( image_shape = (1,chs[2],ac[2],ac[2]), filter_shape = (chs[1],chs[2],conv[2],conv[2]), 
                                poolsize = pool[2],W =layer2_w, b = layer2_b,
                                activation = activation)
                                
        down_layer1 = CPRStage_Down( image_shape = (1,chs[1],ac[1],ac[1]), filter_shape = (chs[0],chs[1],conv[1],conv[1]), 
                                poolsize = pool[1],W =layer1_w, b = layer1_b,
                                activation = activation)
                                
        down_layer0 = CPRStage_Down( image_shape = (1,chs[0],ac[0],ac[0]), filter_shape = (NUM_C,chs[0],conv[0],conv[0]), 
                                poolsize = pool[0],W = layer0_w, b = layer0_b,
                                activation = activation)                                              

        self.Stages = [ up_layer0, up_layer1, up_layer2, up_layer3, up_layer4,
                           down_layer4, down_layer3, down_layer2, down_layer1, down_layer0]
        
    def DeConv( self, input, kernel_index, which_layer=2 ):

        assert kernel_index != None        
        #print "Initial Input Shape", np.shape(input)
        l0u , sw0 = self.Stages[0].GetOutput( input.reshape(1,NUM_C,MAX_PIXEL,MAX_PIXEL) )
        l1u  , sw1 = self.Stages[1].GetOutput( l0u )
        l2u  , sw2 = self.Stages[2].GetOutput( l1u )
        l3u  , sw3 = self.Stages[3].GetOutput( l2u )
        l4u  , sw4 = self.Stages[4].GetOutput( l3u )               
        # only visualize selected kernel - set other output map to zeros
        if which_layer == 4:
            print 'Dimension of l4d', l4u.shape # (1, 32, 2, 2) for example
            l4u[0,:kernel_index,...]*=0
            l4u[0,kernel_index+1:,...]*=0
            
            l4d = self.Stages[5].GetOutput( l4u, sw4 )             
            l3d = self.Stages[6].GetOutput( l4d, sw3 )            
            l2d = self.Stages[7].GetOutput( l3d, sw2 )
            l1d = self.Stages[8].GetOutput( l2d, sw1 )
            l0d = self.Stages[9].GetOutput( l1d , sw0 )               
        elif which_layer == 3:
            l3u[0,:kernel_index,...]*=0
            l3u[0,kernel_index+1:,...]*=0
    
            l3d = self.Stages[6].GetOutput( l3u, sw3 )            
            l2d = self.Stages[7].GetOutput( l3d, sw2 )
            l1d = self.Stages[8].GetOutput( l2d, sw1 )
            l0d = self.Stages[9].GetOutput( l1d , sw0 )       
        
        elif which_layer == 2:
            l2u[0,:kernel_index,...]*=0
            l2u[0,kernel_index+1:,...]*=0
            
            l2d = self.Stages[7].GetOutput( l2u, sw2 )
            l1d = self.Stages[8].GetOutput( l2d, sw1 )
            l0d = self.Stages[9].GetOutput( l1d , sw0 )
        elif which_layer == 1:
            l1u[0,:kernel_index,...]*=0
            l1u[0,kernel_index+1:,...]*=0
            
            #l2d = self.Stages[3].GetOutput( l2u, sw2 )
            l1d = self.Stages[8].GetOutput( l1u, sw1 )
            l0d = self.Stages[9].GetOutput( l1d , sw0 )
        elif which_layer == 0:
            l0u[0,:kernel_index,...]*=0
            l0u[0,kernel_index+1:,...]*=0
            
            
            #l2d = self.Stages[3].GetOutput( l2u, sw2 )
            #l1d = self.Stages[4].GetOutput( l2d, sw1 )
            l0d = self.Stages[9].GetOutput( l0u , sw0 )
        
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
        if which_layer == 4:
            activate_value = Net.forward4(sam)
        elif which_layer == 3:
            activate_value = Net.forward3(sam)
        elif which_layer == 2:
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
        dimx = activate_value.shape[2]
        dimy = activate_value.shape[3]
        newActivateValues = np.zeros((numKernels)) # collapsed version
        #newActivateValues = activate_value.flatten() # not really working for this framework 
        for i in range(numKernels):
            # using the norm to determine the activate value
            newActivateValues[i] = np.linalg.norm(activate_value[0,i])
            #newActivateValues[i] = np.mean(activate_value[0,i])
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

def loadSamplePlanktons(numSamples=100, rotate=False, dim=28):
    if dim == 28:
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
    elif dim == 40:
        from pylearn2_plankton.plankton40Small import PlanktonData
        ds = PlanktonData(which_set='train')
        designMatrix = ds.get_data()[0] # index 1 is the label
        print "Shape of Design Matrix", np.shape(designMatrix)
        designMatrix = np.reshape(designMatrix, 
                                  (ds.get_num_examples(), 1, 40, 40) )
        if numSamples != 'All':
            return np.array(designMatrix[:numSamples,...], dtype=np.float32)
        else:
            return np.array(designMatrix, dtype=np.float32)



def Find_plankton(model_name="plankton_conv_visualize_model.pkl.params", rotate=False, bScalar=False, 
                  start=0, end=16, which_layer=2,
                  numSamples=1000):
    """
    Find plankton that activates the given layers most
    """
    #which_layer = 2
    
    
    samples = loadSamplePlanktons(numSamples=numSamples,rotate=rotate, dim = 40)
    print 'Dimension of Samples', np.shape(samples)
    Net = DeConvNet(model_name, bScalar)
    
    #kernel_list = [ 2,23,60,12,45,9 ]
    kernel_list = range(start,end)
    
    
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
            print 'this_map dimension', this_map.shape
            print 'this_pairmap dimension', this_pairmap.shape
            print 'segment_line dimension', segment_line.shape
        else:
            bigbigmap = np.append(bigbigmap, segment_line, axis = 1)            
            bigbigmap = np.append(bigbigmap, this_pairmap, axis = 1)
            
            
    plt.imshow(bigbigmap)
    plt.show()

yaml_directories = '../../yaml_configs/'
def yamlReadder():
    import yaml
    stream = open("example.yaml", 'r')
    print yaml.load(stream)

if __name__ == "__main__":
    print 'Example: python plankton_vis2.py -rotate -start 16 -end 32 -layer 1'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-rotate', action="store_true", default=True)
    parser.add_argument('-paramsname', action="store", default='small/planktonSmall_0_2.pkl.params')
    parser.add_argument('-start', action="store", default=0, type=int)
    parser.add_argument('-end', action="store", default=16, type=int)
    parser.add_argument('-layer', action="store", default=4, type=int)
    parser.add_argument('-bScalar', action="store_true", default=True) # set default to false after fixing
    results = parser.parse_args()
    rotate = results.rotate
    model_name = results.paramsname
    start = results.start
    end = results.end
    layer = results.layer
    bScalar = results.bScalar
    print 'Loading Params from', model_name
    Find_plankton(model_name, rotate, bScalar, start, end, layer)
    