'''
Created on Mar 13, 2015

@author: keegankang
'''
import theano
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import pickle, os, inspect
import plankton
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.space import Conv2DSpace
from pylearn2.training_algorithms.learning_rule import Momentum

current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
_ , _, _, classDict =  pickle.load( open(os.path.join(current_folder_path,
                                                      '../../data/planktonTrain.p'), 'rb'))

# 1. Create data class / object
class PlanktonData(DenseDesignMatrix):
    def __init__(self):
        # 1. Set class names
        numClasses = 121  
        classNames = []
        for i in range(numClasses):
            classNames.append(classDict[i][0])
        self.class_names = classNames
        
        # 2. Import data 
        train_d, cv_d, test_d = plankton.loadDataSplitted_LeNetFormat()
        X = train_d[0]
        Y = train_d[1]
        Y_oneHot = np.zeros((np.shape(X)[0], numClasses))
        for i, val in enumerate(Y):
            Y_oneHot[i,val] = 1.0
        super(PlanktonData, self).__init__(X=X, y=Y_oneHot)
ds = PlanktonData()

# 2. Create convolutional layer
print 'Creating network layers'
layerh2 = mlp.ConvRectifiedLinear(layer_name='h2', 
                                 output_channels= 64,
                                 irange= .05,
                                 kernel_shape= [5, 5],
                                 pool_shape= [4, 4],
                                 pool_stride= [2, 2],
                                 max_kernel_norm= 1.9365)

layerh3 = mlp.ConvRectifiedLinear(layer_name = 'h3',
                                  output_channels= 64,
                                  irange= .05,
                                  kernel_shape= [5, 5],
                                  pool_shape= [4, 4],
                                  pool_stride= [2, 2],
                                  max_kernel_norm= 1.9365)
''' Note: changed the number of classes '''
layery = mlp.Softmax(max_col_norm = 1.9365,
                     layer_name = 'y',
                     n_classes = 121,
                     istdev = .05)
print 'Setting up trainers'
trainer = sgd.SGD(learning_rate=0.5, batch_size=50, termination_criterion=EpochCounter(200),
                  learning_rule= Momentum(init_momentum=0.5)
                  )
layers = [layerh2, layerh3, layery]
ann = mlp.MLP(layers, input_space=Conv2DSpace(shape= [28, 28], num_channels=1))
trainer.setup(ann, ds)
print 'Start Training'
while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    if not trainer.continue_learning(ann):
        break

# 3. Predict
XReport, Y_info = plankton.loadReportData()
probMatrix = ann.fprop(theano.shared(XReport, name='XReport')).eval()
plankton.generateSubmissionFile(probMatrix, classDict, Y_info)