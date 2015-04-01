'''
Created on Mar 30, 2015

@author: ben

This file takes a model file (.pkl) and visualize the convolutional layers

Instructions:
python visualizeLayers.py [path to model file]
Example:
python visualizeLayers.py model_files/plankton_conv_model.pkl
'''
from pylearn2.utils import serial
from pylearn2.gui import get_weights_report
from pylearn2.config import yaml_parse
import numpy as np

def showWeights(model_name, model, rescale="individual", border=None, out=None):
    pv = get_weights_report.get_weights_report(model_path=model_name,
                                               rescale=rescale,
                                               border=border)

    if out is None:
        pv.show()
    else:
        pv.save(out)
        
def showWeightsBinocular(model_name, model, rescale="individual", border=None, out=None):
    pv = get_weights_report.get_binocular_greyscale_weights_report(model_path=model_name,
                                               rescale=rescale,
                                               border=border)

    if out is None:
        pv.show()
    else:
        pv.save(out)

def visualizeConvolutionalLayers3():
    pass


'''
Understand the functions defined on model. 
Understand the weights
'''
def inspectModel(model):
    weights_topo = model.get_weights_topo()
    print 'model.get_weights_topo', np.shape(weights_topo)
    # (64, 5, 5, 1) - what does this mean?
    print 'model.get_weights_format', model.get_weights_format()
    print 'data set', model.dataset_yaml_src
    #print 'Loading Dataset'
    #dataset = yaml_parse.load(model.dataset_yaml_src)

def main():
    import sys
    try:
        model_name = sys.argv[1]
        print 'Loading', model_name
        model = serial.load(model_name)
    except IndexError:
        print 'Please specify model name in the argument. Eg. python visualizeLayers.py model_files/plankton_conv_model.pkl'
    inspectModel(model)
    showWeights(model_name, model)

if __name__ == "__main__":
    main()