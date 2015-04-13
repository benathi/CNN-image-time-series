'''
Created on Apr 12, 2015

@author: ben
'''
import numpy as np
from plankton_vis2 import DeConvNet
import os, sys, pickle
from ggplot import *
import pandas as pd


# aux libraries
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
class_path = parent_folder + '/' + 'utils'
if class_path not in sys.path:
    sys.path.append( class_path )
import plankton_utils

def findActivations(Cnn, samples,
                    which_layer=2):
    print "Finding Activation Values at Layer", which_layer
    activations_filename = 'activations.npy'
    if os.path.isfile(activations_filename):
        print 'Loading Activations from files'
        return np.load(file(activations_filename, 'r'))
    
    # Construct activations matrix
    numSamples = np.shape(samples)[0]
    first_sample = samples[0].reshape((1,) + samples[0].shape )  
    
    if which_layer == 2:
        activations = np.zeros((numSamples,) + np.shape(Cnn.forward2(first_sample).flatten()) )
    elif which_layer == 1:
        activations = np.zeros((numSamples,) + np.shape(Cnn.forward1(first_sample).flatten()) )
    elif which_layer == 2:
        activations = np.zeros((numSamples,) + np.shape(Cnn.forward0(first_sample).flatten()) )
    print 'Shape of activations matrix', np.shape(activations)
    for i, sam in enumerate(samples):
        if i % 500 == 0:
            print 'Sample No', i
        sam = sam.reshape((1,) + sam.shape )   
        if which_layer == 2:
            activate_value = Cnn.forward2(sam)
        elif which_layer == 1:
            activate_value = Cnn.forward1(sam)
        elif which_layer == 0:
            activate_value = Cnn.forward0(sam)
        
        activations[i] = activate_value.flatten() # note: first axis is simply batch size
    np.save(open(activations_filename, 'w'), activations)
    return activations

def separateActivationsByClass(activations, y_labels):
    ''' Initialization '''
    label_count = [0]*121
    for y_label in y_labels:
        label_count[y_label] += 1
    activation_dict = {}
    for class_num in range(121):
        _size = label_count[class_num]
        activation_dict[class_num] = np.zeros( (_size,) + np.shape(activations)[1:])
    
    '''  '''
    label_count = [0]*121
    for i, y_label in enumerate(y_labels):
        #print 'i= %d.\ty_label = %d' % (i, y_label)
        #print 'Shape of activation_dict[y_label]', np.shape(activation_dict[y_label])
        activation_dict[y_label][label_count[y_label]] = activations[i]
        #######
        label_count[y_label] += 1
    ''' TODO - flatten to 1D for each one'''
    #print "shape of class 0", np.shape(activation_dict[0]) # - looks correct
    return activation_dict

def calculateAverageActivations(activation_dict, which_layer,savePlots=False):
    # assume that each activation has been flatten to vector
    numFeatures = np.shape(activation_dict[0])[1]
    print 'num features =', numFeatures
    averageActivations = np.empty((len(activation_dict.keys()),
                                   numFeatures))
    print 'Shape of average activations', np.shape(averageActivations)
    
    for class_num in activation_dict:
        #print averageActivations[class_num,:].shape
        #print np.mean(activation_dict[class_num], axis=0).shape
        averageActivations[class_num,:] = np.mean(activation_dict[class_num], axis=0) + 0.00001
        print 'class_num', class_num
        if savePlots:
            print averageActivations[class_num]
        df = pd.DataFrame({"x": range(numFeatures), "y":averageActivations[class_num],
                           "feature_num":range(numFeatures), "positions":[0.0]*numFeatures})
        _plot = ggplot(aes(x="x", y="y"), df) + geom_bar(fill='#9ac37b', stat="bar") \
            + ggtitle('Average Activation for Class ' + str(class_num) + ' Layer' + str(which_layer) ) \
            + xlab('Feature') + ylab('Activation') \
            + geom_text(aes(label='feature_num', y='y', size=8, vjust=0.05, hjust=-0.3))
            #+ scale_x_continuous(breaks=range(0,numFeatures))
        if savePlots:
            ggsave(_plot, 'results/averageActivationLayer' + str(which_layer) + 'Class' + str(class_num) + '.png', 
                   width=16, height=6)
        #print _plot
        #break
    np.save(open('average_activationsLayer' + str(which_layer) + '.npy', 'w'), averageActivations)

def getAverageActivationsMatrix(which_layer=2):
    return np.load(open('average_activationsLayer' + str(which_layer) + '.npy', 'r'))
    
def main(model_name="plankton_conv_visualize_model.pkl.params",
         rotate=False, bScalar=True, 
         which_layer=2, numSamples='All'):
    import plankton_vis1
    samples = plankton_vis1.loadSamplePlanktons(numSamples, rotate)
    print 'Dimension of Samples', np.shape(samples)
    Cnn = DeConvNet(model_name, bScalar)
    activations = findActivations(Cnn, samples, which_layer)
    y_labels = plankton_utils.getY_label_int(which_data='train')
    activation_dict = separateActivationsByClass(activations, y_labels)
    calculateAverageActivations(activation_dict, which_layer)
    

if __name__ == "__main__":
    main()