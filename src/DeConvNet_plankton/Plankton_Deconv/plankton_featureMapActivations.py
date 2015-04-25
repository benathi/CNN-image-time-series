'''
Created on Apr 12, 2015

@author: ben

TODO: Should we use flatten to unpack feature map?
Can we use the maximum value?
Flatten preserves locality
np.array([  [[1,2],[3,4]], [[5,6],[7,8]] ]).flatten()
array([1, 2, 3, 4, 5, 6, 7, 8])

Take maximum for feature map? or average?
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

def findActivations(model_name, samples,
                    which_layer=2, allFeats=False, bScalar=True, cache=False):
    print "Finding Activation Values at Layer", which_layer
    if cache:
        if allFeats:
            activations_filename = 'activations_layer' + str(which_layer) + '_allFeats'  +'.npy'
        else:
            activations_filename = 'activations_layer' + str(which_layer) +'.npy'
        if os.path.isfile(activations_filename):
            print 'Loading Activations from files'
            return np.load(file(activations_filename, 'r'))
    Cnn = DeConvNet(model_name, bScalar)
    # Construct activations matrix
    numSamples = np.shape(samples)[0]   # 
    print 'Shape of sample 0 is', samples[0].shape
    first_sample = samples[0].reshape((1,) + samples[0].shape )  
    
    if which_layer == 2:
        activations = np.zeros((numSamples,) + np.shape(Cnn.forward2(first_sample).flatten()) )
    elif which_layer == 1:
        if allFeats:
            activations = np.zeros((numSamples,) + np.shape(Cnn.forward1(first_sample).flatten()) )
        else:
            activations = np.zeros((numSamples,) + (np.shape(Cnn.forward1(first_sample))[1],) )
    elif which_layer == 0:
        if allFeats:
            activations = np.zeros((numSamples,) + np.shape(Cnn.forward0(first_sample).flatten()) )
        else:
            activations = np.zeros((numSamples,) + (np.shape(Cnn.forward0(first_sample))[1],) )
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
        if i == 0:
            print 'Activation Shape =', activate_value.shape
            # (1, 48, 4, 4) for layer 1 for example
        if allFeats:
            activations[i] = activate_value.flatten() # note: first axis is simply batch size
        else:
            activations[i] = np.mean(activate_value, axis=(2,3))
    if cache:
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

def calculateAverageActivations(activation_dict, which_layer,savePlots=True):
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
            #print averageActivations[class_num]
            pass
        df = pd.DataFrame({"x": range(numFeatures), "y":averageActivations[class_num],
                           "feature_num":range(numFeatures), "positions":[0.0]*numFeatures})
        _plot = ggplot(aes(x="x", y="y"), df) + geom_bar(fill='#9ac37b', stat="bar") \
            + ggtitle('Average Activation for Class ' + str(class_num) + ' Layer' + str(which_layer) ) \
            + xlab('Feature') + ylab('Activation') \
            + geom_text(aes(label='feature_num', y='y', size=8, vjust=0.0, hjust=-0.3))
            #+ scale_x_continuous(breaks=range(0,numFeatures))
        if savePlots:
            ggsave(_plot, 'results/activationsVis/layer' + str(which_layer)
                   + '/averageActivationLayer' + str(which_layer)
                   + 'Class' + str(class_num) + '.pdf',
                   width=16, height=6)
        #print _plot
        #break
    np.save(open('average_activationsLayer' + str(which_layer) + '.npy', 'w'), averageActivations)

''' Helper method for other scripts to use '''
def getAverageActivationsMatrix(which_layer=2, allFeats=False):
    if allFeats:
        return np.load(open('average_activationsLayer' + str(which_layer)  + '_allFeats' + '.npy', 'r'))
    else:
        return np.load(open('average_activationsLayer' + str(which_layer) + '.npy', 'r'))

''' A method to plot all activations as lines '''
def plotFeatures(featuresDict, which_layer=2, savePlots=True):
    print 'shape of featuresDict[0]', featuresDict[0].shape
    numFeatures = np.shape(featuresDict[0])[1]
    for class_num in featuresDict:
        print 'class', class_num
        filename = 'results/allActivationsVis/layer' + str(which_layer) \
            + '/activationLayer' + str(which_layer) \
            + 'Class' + str(class_num) + '.pdf'
        num_points = featuresDict[class_num].shape[0]
        
        bigy = featuresDict[class_num].flatten()
        bigx = range(numFeatures)*num_points
        bigvar = []
        for _k in range(num_points):
            bigvar += [_k]*numFeatures
            # maybe use string
        
        df = pd.DataFrame({"x":bigx,
                           "y":bigy,
                           "sample":bigvar
                           })
        _plot = ggplot(aes(x='x', y='y', color='sample'), df) + xlab('Feature') + ylab('Activation') \
        + ggtitle('Average Activation for Class' + str(class_num) + ' Layer' + str(which_layer) ) \
        + geom_line() + geom_point()
        
        '''
        for _j in range(num_points):
            print 'add plot ', _j
            _df = pd.DataFrame({"x":range(numFeatures),
                                "y":featuresDict[class_num][_j]
                                })
            _plot + geom_point(data=_df)
        '''
        #print _plot
        if savePlots:
            ggsave(_plot, filename, width=16, height=6)
        
    
def main(model_name="plankton_conv_visualize_model.pkl.params",
         rotate=False, bScalar=True, 
         which_layer=2, numSamples='All'):
    import plankton_vis1
    samples = plankton_vis1.loadSamplePlanktons(numSamples, rotate)
    print 'Dimension of Samples', np.shape(samples)
    activations = findActivations(model_name, samples, which_layer, 
                                  allFeats=False, bScalar=False)
    y_labels = plankton_utils.getY_label_int(which_data='train')
    activation_dict = separateActivationsByClass(activations, y_labels)
    #calculateAverageActivations(activation_dict, which_layer)
    plotFeatures(activation_dict, which_layer)
    

if __name__ == "__main__":
    main()