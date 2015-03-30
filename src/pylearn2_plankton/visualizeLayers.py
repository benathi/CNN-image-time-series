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

def showWeights(model_name, model, rescale="individual", border=None, out=None):
    pv = get_weights_report.get_weights_report(model_path=model_name,
                                               rescale=rescale,
                                               border=border)

    if out is None:
        pv.show()
    else:
        pv.save(out)

def visualizeConvolutionalLayers():
    pass


def main():
    import sys
    try:
        model_name = sys.argv[1]
        print 'Loading', model_name
        model = serial.load(model_name)
        showWeights(model_name, model)
    except IndexError:
        print 'Please specify model name in the argument. Eg. python visualizeLayers.py model_files/plankton_conv_model.pkl'

if __name__ == "__main__":
    main()