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
        if updated_model['layers'][l]['params']['W'] is not None:
            updated_model['layers'][l]['params']['W'] -= a * (grads[l]['W'] - lmd * updated_model['layers'][l]['params']['W'])
        if updated_model['layers'][l]['params']['b'] is not None:
            updated_model['layers'][l]['params']['b'] = a * (grads[l]['b'] - lmd * updated_model['layers'][l]['params']['b'])

    return updated_model