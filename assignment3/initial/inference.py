import numpy as np

def inference(model, input):
    """
    Do forward propagation through the network to get the activation
    at each layer, and the final output
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [batch_size]
    Returns:
        output: The final output of the model
        activations: A list of activations for each layer in model["layers"]
    """

    num_layers = len(model['layers'])
    activations = [None,] * num_layers

    # TODO: FORWARD PROPAGATION CODE
    for l in range(num_layers):
        layer = model['layers'][l]
        activations[l], _, _ = layer['fwd_fn'](input, layer['params'], layer['hyper_params'], False)
        input = activations[l]

    output = activations[-1]
    return output, activations
