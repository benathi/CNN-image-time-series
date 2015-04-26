'''
Created on Apr 25, 2015

@author: ben

A class for configurable deconvolutional network
'''

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
        
        '''
        Note: after applying convolution, our weight is a weight per pixel
        whereas the weight here is per the whole array. Fix this by averaging for now
        '''
        if bScalar:
            print "Using b as scalar"
            layer0_b = np.mean(layer0_b, axis=(1,2))
            layer1_b = np.mean(layer1_b, axis=(1,2))
            layer2_b = np.mean(layer2_b, axis=(1,2))
        else:
            print "Using b as matrix"

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