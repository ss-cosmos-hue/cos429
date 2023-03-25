"""
Basic script to create a new network model.
The model presented here is meaningless, but it shows how to properly 
call init_model and init_layers for the various layer types.
"""

import sys
sys.path += ['pyc_code', 'layers']
import numpy as np
from init_layers import init_layers
from init_model import init_model
from inference import inference
from inference_ import inference as inference_
from loss_euclidean import loss_euclidean

def pass_fail(cond):
    if cond:
        str = 'Passed!'
    else:
        str = 'Failed.'
    return str

def mse(val1, val2):
    assert val1.shape == val2.shape
    return np.sum((val1 - val2) ** 2) / val1.size

def main():
    np.random.seed(0)
    err_thresh = 1e-6

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

    # Example calls you might make for inference:
    inp = np.random.rand(10, 10, 3, 3)    # Dummy input
    out, _ = inference(model, inp)
    out_, _ = inference_(model, inp)

    print(out)
    print(out_)

    print('Inference')
    if out is None:
        print('\tout DNE')
        print('\tTest: %s' % pass_fail(False))
    else:
        err_out = mse(out, out_)
        if not (out.shape==out_.shape):
            print('\tout size does not match!')
            print('\tTest: %s' % pass_fail(False))
        else:
            print('\tTest: %s' % pass_fail(err_out < err_thresh))
    
    
    
    # Example calls you might make for calculating loss:
    output = np.random.rand(10, 20)       # Dummy output
    ground_truth = np.random.rand(10, 20) # Dummy ground truth
    loss, _ = loss_euclidean(output, ground_truth, {}, False)

if __name__ == '__main__':
    main()
