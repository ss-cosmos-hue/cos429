import numpy as np

def update_weights(model, grads, hyper_params):
    '''
    Update the weights of each layer in your model based on the calculated gradients
    Args:
        model: Dictionary holding the model
        grads: A list of gradients of each layer in model["layers"]
        hyper_params: 
            hyper_params['learning_rate']
            hyper_params['weight_decay']: Should be applied to W only.
    Returns: 
        updated_model:  Dictionary holding the updated model
    '''
    num_layers = len(grads)
    a = hyper_params["learning_rate"]
    lmd = hyper_params["weight_decay"]
    
    updated_model = model

    # TODO: Update the weights of each layer in your model based on the calculated gradients
    for l in range(num_layers):
        layer = model['layers'][l]
        if layer['type']=='linear' or layer['type']=='conv':
            layer['params']['W'] -= a * (grads[l]['W'] + lmd * layer['params']['W'])
            layer['params']['b'] -= a * grads[l]['b']

    return updated_model