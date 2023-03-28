import numpy as np

def update_avegrads(model,grads, avegrads,rho=0.99):
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
    # TODO: Update the weights of each layer in your model based on the calculated gradients
    
    for i in range(num_layers):
        layer = model['layers'][i]
        if layer['type']=='linear' or layer['type']=='conv':
            avegrads[i]['W']=avegrads[i]['W']*rho + grads[i]['W']
            avegrads[i]['b']=avegrads[i]['b']*rho + grads[i]['b']
    return avegrads