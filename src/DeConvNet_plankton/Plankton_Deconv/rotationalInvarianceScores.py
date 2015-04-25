'''
Created on Apr 24, 2015

@author: ben

This script is to plot rotational invariant scores for each feature / layer
'''
import numpy as np
import pandas as pd
from ggplot import *
import skimage
from skimage.viewer import ImageViewer
from matplotlib import pyplot as plt
import os,sys
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
class_path = parent_folder + '/' + 'utils'
if class_path not in sys.path:
    sys.path.append( class_path )

from generateRotatedData import generateRotatedData # part of utils

def generateRotatedMinibatch():
    import plankton_vis1
    samples = plankton_vis1.loadSamplePlanktons(100) # 100,1,28,28
    if False:
        for i, sample in enumerate(samples):
            print 'Showing sample ',i
            plt.imshow(sample)
            plt.show()
    rotationalSym_indicies = [0,3,6,16] # just to keep track of which index we can use
    
    chosen_plank = samples[[0]]
    rotatedSamples = generateRotatedData(chosen_plank,[0])
    #print 'rotated samples', rotatedSamples
    print 'shape of rotated samples', np.shape(rotatedSamples[0])
    return rotatedSamples[0] # only the X

def calculateInvarianceScores(model_name, which_layer):
    from plankton_featureMapActivations import findActivations
    samples = generateRotatedMinibatch()
    print 'shape of samples before findActivations', samples.shape
    findActivations(model_name, samples,
                    which_layer=2, allFeats=True, cache=False)
    ''' output a dictionary '''

def plotRinvariance(featuresDict, which_layer=2, savePlots=True):
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

def plotInvarianceScores_main(model_name, which_layer=2):
    rInvarianceScores = calculateInvarianceScores(model_name, which_layer)
    plotRinvariance(rInvarianceScores, which_layer)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-layer', action="store", default=2, type=int)
    parser.add_argument('-paramsname', action="store", default='plankton_conv_visualize_model.pkl.params')
    allArgs = parser.parse_args()
    layer = allArgs.layer
    model_name = allArgs.paramsname
    plotInvarianceScores_main(model_name, which_layer=layer)

if __name__ == '__main__':
    main()