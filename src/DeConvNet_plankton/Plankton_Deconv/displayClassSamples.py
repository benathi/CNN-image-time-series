'''
Created on Apr 12, 2015

@author: ben

This script saves 4 images in grid for each class

11 x 11 of (2,2) grid
'''
import numpy as np
import os, sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from plankton_vis1 import loadSamplePlanktons

parent_folder = os.path.abspath('..')
class_path = parent_folder + '/' + 'DeConvNet'
if class_path not in sys.path:
    sys.path.append( class_path )
class_path = parent_folder + '/' + 'utils'
if class_path not in sys.path:
    sys.path.append( class_path )
    
from utils import tile_raster_images
import plankton_utils
MAX_PIXEL = 28
NUM_C = 1

def displayClassSamples(saveIndividualSamples=False):
    # from bc01 to cb01
    samples = loadSamplePlanktons(30000, rotate=False)
    print 'samples shape', samples.shape #e.g. (30000,1,28,28)
    y_labels = plankton_utils.getY_label_int(which_data='train')[:30000]
    # construct list of indices for each label
    class_indices = [None]*121
    for _index, y_label in enumerate(y_labels):
        #print "label %d _index = %d" % (y_label, _index)
        if class_indices[y_label] == None:
            class_indices[y_label] = [_index]
        else:
            class_indices[y_label].append(_index);
        #print class_indices[y_label]
    
    for i, this_class_indices in enumerate(class_indices):
        if len(this_class_indices) < 4:
            print 'Warning! Increase sample size! (Found less than 4 samples for class ', i
            return
    
    if saveIndividualSamples:
        ''' Saving sample image for each class '''
        #print 'this_class_indices', this_class_indices
        shape = samples[this_class_indices[0]].shape
        #print 'shape of sample', shape    #e.g. (1,28,28)
        for class_num, this_class_indices in enumerate(class_indices):
            tile4 = None
            for i in range(2):
                tile2 = None
                for j in range(2):
                    if tile2 == None:
                        tile2 = np.reshape(samples[this_class_indices[2*i+j]], (shape[1],shape[2]))
                    else:
                        tile2 = np.concatenate( (tile2, 
                                             np.reshape(samples[this_class_indices[2*i+j]], (shape[1],shape[2]))
                                             ),
                                            axis=0)
                if tile4 == None:
                    tile4 = tile2
                else:
                    tile4 = np.concatenate( (tile4, tile2), axis=1)
            print 'class_num', class_num ,'shape of tile4', tile4.shape
            fname = 'results/classSamples_grey/' + str(class_num) + ".png"
            if False:
                from PIL import Image
                newIm = 255.*(tile4)/2. + 127.
                im = Image.fromarray(newIm)
                im = im.convert('RGB')
                print 'Saving class num=', class_num
                im.save(fname)
            else:
                plt.imshow(newIm, cmap = cm.Greys_r)
                plt.axis('off')
                plt.show()
                plt.savefig(fname)
    
    
    
    
    bigbigmap = None
    for row in range(11):
        bigmap = None
        for col in range(11):
            class_index = 11*row+col
            print 'Class Index', class_index
            print "Indices with this class", class_indices[class_index][:4]
            class_samples = samples[class_indices[class_index][:4]]
            print "class_samples shape", class_samples.shape
            
            '''
            for pairs in heap:
                this_sam = pairs.sam
                this_sams.append( this_sam.reshape([NUM_C,MAX_PIXEL,MAX_PIXEL]) )
                this_Deconv.append( Net.DeConv( this_sam, kernel_index, which_layer=which_layer ).reshape([NUM_C,MAX_PIXEL,MAX_PIXEL]) )
            '''
            
            this_sams = np.transpose( class_samples, [ 1, 0, 2, 3 ])
            this_sams = this_sams.reshape( [ NUM_C, 4, MAX_PIXEL*MAX_PIXEL ])
            this_sams = tuple( [ this_sams[0] for i in xrange(3)] + [None] )    
            
            this_map = tile_raster_images( class_samples, img_shape = (MAX_PIXEL,MAX_PIXEL), tile_shape = (2,2), 
                                       tile_spacing=(1, 1), scale_rows_to_unit_interval=True, 
                                        output_pixel_vals=True)
    
            
            if bigmap == None:
                bigmap = this_map
                segment_line = 255*np.ones([bigmap.shape[0],1],dtype='uint8')
                print 'this_map dimension', this_map.shape
                print 'segment_line dimension', segment_line.shape
            else:
                bigmap = np.append(bigmap, segment_line, axis = 1)            
                bigmap = np.append(bigmap, this_map, axis = 1)
        if bigbigmap == None:
            bigbigmap = bigmap
        else:
            bigbigmap = np.append(bigbigmap, 255*np.ones((1, bigmap.shape[1]), dtype='uint8'), axis=0)
            bigbigmap = np.append(bigbigmap, bigmap, axis=0)
    print 'shape of big big map', bigbigmap.shape
    plt.imshow(bigbigmap,cmap = cm.Greys_r)
    plt.xticks(np.arange(11))
    plt.yticks(np.arange(11))
    plt.show()

if __name__ == '__main__':
    displayClassSamples()
    # note: result saved