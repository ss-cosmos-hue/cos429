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
from loss_euclidean import loss_euclidean
from inference import inference
from inference_ import inference as inference_
from calc_gradient import calc_gradient
from calc_gradient_ import calc_gradient as calc_gradient_
from update_weights import update_weights
from update_weights_ import update_weights as update_weights_

def pass_fail(cond):
    if cond:
        str = 'Passed!'
    else:
        str = 'Failed.'
    return str

def mse(val1, val2):
    if type(val1) == np.ndarray:
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

    # Test for inference
    inp = np.random.rand(10, 10, 3, 3)
    out, acts = inference(model, inp)
    out_, acts_ = inference_(model, inp)

    print('Inference')
    print('\tTest: %s' % pass_fail(mse(out, out_) < err_thresh))

    # Test for calc_gradient
    dv_output = np.random.rand(len(out), len(out[0]))
    grads = calc_gradient(model, inp, acts, dv_output)
    grads_ = calc_gradient_(model, inp, acts_, dv_output)

    print('Calculate Gradient')
    print('\tTest: %s' % pass_fail(mse(grads[0]['W'], grads_[0]['W']) < err_thresh))
    
    # Test for update_weights
    hyper_params = {'learning_rate' : 1e-3, 'weight_decay' : 1e-3}
    new_model = update_weights(model, grads, hyper_params)
    new_model_ = update_weights(model, grads_, hyper_params)
    
    print('Update Weights')
    print('\tTest: %s' % pass_fail(mse(new_model['layers'][0]['params']['W'], new_model_['layers'][0]['params']['W']) < err_thresh))

if __name__ == '__main__':
    main()
