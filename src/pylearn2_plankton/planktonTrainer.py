'''
Created on Mar 13, 2015

@author: keegankang
'''
from pylearn2.config import yaml_parse
import os,sys
import DeepLearning_Utils



projectPath = DeepLearning_Utils.getCurrentPath()
sys.path.append(projectPath)
print sys.path


filePath = os.path.join(projectPath, 'pylearn2_plankton', 'plankton_conv.yaml')
print filePath
train = open(filePath,'r').read()
train = yaml_parse.load(train)
train.main_loop()