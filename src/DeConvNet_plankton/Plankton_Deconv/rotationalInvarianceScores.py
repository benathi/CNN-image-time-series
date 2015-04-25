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
import matplotlib.cm as cm
import os,sys
parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
class_path = parent_folder + '/' + 'utils'
if class_path not in sys.path:
    sys.path.append( class_path )

from generateRotatedData import generateRotatedDataPreserveDim # part of utils


def choose_samples(indices=[0,3,6,16]):
    import plankton_vis1
    samples = plankton_vis1.loadSamplePlanktons(20) # 100,1,28,28
    if False:
        for i, sample in enumerate(samples):
            print 'Showing sample ',i
            plt.imshow(sample[0], cmap = cm.Greys_r)
            plt.show()
    return samples[indices]

def generateRotatedMinibatch(samples, indicies, numRotations):
    # just to keep track of which index we can use
    ''' The samples has shape (100,1,28,28) for example'''
    print 'at rotated minibatch: shape of samples', samples.shape
    rotatedSamples = generateRotatedDataPreserveDim(samples, indicies)
    print 'shape of rotated samples', np.shape(rotatedSamples[0])
    return rotatedSamples # only the X


'''
acts:    activation matrix
        (num rotations, numFeatures)
'''
def rInvScore(acts):
    num_feats = acts.shape[1]
    print 'Number of features', num_feats
    #print 'Activations = ', acts
    scores = np.zeros((num_feats))
    for feat_idx in range(num_feats):
        _min = np.min(acts[:,feat_idx])
        _max = np.max(acts[:,feat_idx])
        scores[feat_idx] = max(0,_min)/_max
        if _max == 0:
            scores[feat_idx] = 0
    return scores

def calculateInvarianceScores(samples, indices, model_name, which_layer, numRotations=12):
    from plankton_featureMapActivations import findActivations
    rotatedSamples, sample_indices = generateRotatedMinibatch(samples, indices,
                                                              numRotations)
    print 'shape of samples before findActivations', samples.shape
    activations = findActivations(model_name, rotatedSamples,
                    which_layer=2, allFeats=True, cache=False)
    print 'shape of activations', activations.shape
    ''' output a dictionary '''
    dictSampleRotation = {}
    for num, idx in enumerate(indices):
        dictSampleRotation[idx] = activations[num*numRotations:(num+1)*numRotations]
    
    # format: key = sample_index, value = [rinv_0, r_inv1, ...]
    rInvScoreDict = {}
    for key in dictSampleRotation:
        rInvScoreDict[key] = rInvScore(dictSampleRotation[key])
    
    #print rInvScoreDict
    return rInvScoreDict
    

def plotRinvariance(scoresDict, which_layer=2, savePlots=True):
    numPlots = len(scoresDict.keys())
    numFeatures = scoresDict[0].shape[0] # first 0 is in the key
    idx = scoresDict.keys()
    
    bigx = range(numFeatures)*numPlots
    bigy = []
    for id in idx:
        bigy.append(scoresDict[id])
    bigy = np.array(bigy)
    bigy = bigy.flatten()
    bigkey = []
    for id in idx:
        for n in range(numFeatures):
            bigkey.append(str(id))
    
    print len(bigx)
    print len(bigy)
    print len(bigkey)
    
    df = pd.DataFrame({"x":bigx,
                       "y":bigy,
                       "sample":bigkey
                       })
    plot = (ggplot(aes(x='x', y='y', color='sample'), df) +
    xlab('Feature') + ylab('Rotationally Invariant Scores')
    + ggtitle('Rotationally Invariance Scores of Features')
    #+ geom_line(alpha=0.5) 
    + geom_point(alpha=0.5))
    
    print plot
    # note: issue for legend not showing
    '''
    fig = plot.draw()
    ax = fig.axes[0]
    offbox = ax.artists[0]
    offbox.set_bbox_to_anchor((1, 0.5), ax.transAxes)
    print fig
    '''



def plotInvarianceScores_main(model_name, which_layer=2):
    indices = [0,3,6,16]
    chosen_samples = choose_samples(indices)
    rInvarianceScores = calculateInvarianceScores(chosen_samples, indices,
                                                  model_name, 
                                                  which_layer,
                                                  numRotations=12)
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