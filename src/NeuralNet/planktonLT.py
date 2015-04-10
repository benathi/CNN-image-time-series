
from NeuralNet import NeuralNet
import numpy as np


def main():
    nn = NeuralNet(trainingData = 'PLANKTONTRANSFORM',
                   hiddenLayersSize=[50]*2, 
                 activationFunctions=['sigmoid']*3)
    print [np.shape(ob) for ob in nn.Thetas]
    #nn.train(maxNumIts=10000,regParams=[0.1]*3)
    nn.train_cg(regParams=[0.1]*6, MaxIts=500)
    print '', [np.shape(i) for i in nn.trainData]
    print '', np.sum((nn.data_labels != nn.classify(nn.trainData[0])))/(1.0*nn.numInputs)
    
if __name__ == "__main__":
    #testNeuralNet()
    main()    