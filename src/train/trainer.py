'''
Created on Mar 13, 2015

@author: benathi

This script trains a model based on a yaml config and save a model file at the end.

Instructions:
python trainer.py [path to yaml config]
For example, 
python trainer.py yaml_configs/plankton_conv.yaml
'''
from pylearn2.config import yaml_parse
import os,sys,inspect
import theano
import numpy as np


def train(yaml_filename):
    current_folder_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    filePath = os.path.join(current_folder_path, yaml_filename)
    print 'Reading YAML Configurations'
    trainObj = open(filePath,'r').read()
    print 'Loading Train Model'
    trainObj = yaml_parse.load(trainObj) # serial.load_train_file('--.yaml')
    print 'Looping'
    trainObj.main_loop()
    return trainObj


def trainAndReport():
    import sys
    try:
        yaml_filename = sys.argv[1]
        print 'Loading', yaml_filename
        train(yaml_filename)
    except IndexError:
        print 'Please specify YAML filename in the argument. Eg. python planktonTrainer.py yaml_configs/cifar10.yaml'
    

if __name__=='__main__':
    trainAndReport()