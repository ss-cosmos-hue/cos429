import numpy as np

def calc_gradient(model, input, layer_acts, dv_output):
    '''
    Calculate the gradient at each layer, to do this you need dv_output
    determined by your loss function and the activations of each layer.
    The loop of this function will look very similar to the code from
    inference, just looping in reverse.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
        layer_acts: A list of activations of each layer in model["layers"]
        dv_output: The partial derivative of the loss with respect to each element in the output matrix of the last layer.
    Returns: 
        grads:  A list of gradients of each layer in model["layers"]
    '''
    num_layers = len(model["layers"])
    grads = [None,] * num_layers#in order

    # TODO: Determine the gradient at each layer.
    #       Remember that back-propagation traverses 
    #       the model in the reverse order.
    backprop = True
    assert (num_layers>=2),num_layers
    
    layer = model['layers'][-1]
    _,dv_input,grads[-1] = layer['fwd_fn'](layer_acts[-1],layer['params'],layer['hyperparams'],backprop,dv_output=dv_output)
    
    for i in np.arange(1,num_layers-1)[::-1]:
        layer = model['layers'][i]
        _,dv_input,grads[i] = layer['fwd_fn'](layer_acts[i-1],layer['params'],layer['hyperparams'],backprop,dv_output=dv_input)
    
    layer = model['layers'][0]
    _,dv_input,grads[0] = layer['fwd_fn'](input,layer['params'],layer['hyperparams'],backprop,dv_input)
    

    return grads