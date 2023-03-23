import sys
sys.path += ['layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model

from train import train

def main():
    #todo update model
    l = [init_layers('conv', {'filter_size': 2,
                              'filter_depth': 3,
                              'num_filters': 2}),
         init_layers('pool', {'filter_size': 2,
                              'stride': 2}),
         init_layers('relu', {}),
         init_layers('flatten', {}),
         init_layers('linear', {'num_in': 32,
                                'num_out': 10}),
         init_layers('softmax', {})]

    model = init_model(l, [10, 10, 3], 10, True)

    inp = np.random.rand(10, 10, 3, 3)    # Dummy input
    #dataload
    #create input and label
    
    params =  {"learning_rate":0.01,
               "weight_decay":0.001,
                "batch_size":128,
                "save_file":'model.npz'}
    #train
    model,loss = train(model, input, label, params, numIters=100)
    #to do validation, test 


if __name__ == '__main__':
    main()
