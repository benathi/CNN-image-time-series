'''
Created on Apr 2, 2015

@author: ben

This is a helper script to extract params from model (.pkl) file. This is helpful for 
model run with gpu.
'''
from pylearn2.utils import serial
import cPickle
trainedModelPath = "../../pylearn2_plankton/model_files/"


'''
Usually, a model will be trained on a gpu unit. This method should be run on it after training
to extract numpy array values. (the extraction requires cuda, hence it needs to be run with gpu)
After extracting, it saves the params with cPickle and we can load it directly.
'''
def recordModelParams(model_name = "plankton_conv_visualize_model.pkl",numLayer = 3):
    model_path= trainedModelPath + model_name
    print 'Loading Model'
    model = serial.load(model_path)
    print "Done Loading Model"
    print "Input Space:\t", model.get_input_space()
    print "Target Space:\t", model.get_target_space() # same as get_output_space()
    print "Monitoring Data Specs", model.get_monitoring_data_specs()
    param_names = model.get_params()
    param_names = [tensorVar.name for tensorVar in param_names]
    print "Params Spec", param_names
    layer_names = []
    for i in range(numLayer):
        strname = "c" + str(i) + "_b";
        layer_names.append(strname)
        strname = "c" + str(i) + "_W";
        layer_names.append(strname)   
#    layer_names = ['c2_W', 'c2_b', 'c1_W', 'c1_b', 'c0_W', 'c0_b']
    layer_names.reverse()
    print "type", type(param_names[0])
    print "index of c0_W", param_names.index('c0_W')
    # assume there are 3 layers
    original_params = model.get_param_values()
    params_indices = [param_names.index(_name) for _name in 
                      layer_names]
    print "Parameter Indices for {} is {}".format(layer_names, params_indices)
    params = [original_params[_index] for _index in params_indices]
    cPickle.dump(params,open(model_path + ".params", "wb"))
    
if __name__ == "__main__":
    print "Running extractParamsFromPkl"
    import sys
    try:
        pkl_filename = sys.argv[1]
        numofLayers = int(sys.argv[2])
    except IndexError:
        print 'Please specify model (.pkl) name in the argument and number of layers. Eg. python extractParamsFromPkl.py plankton_conv_visualize_model.pkl 5'
    recordModelParams(pkl_filename, numofLayers)