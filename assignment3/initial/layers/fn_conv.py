import numpy as np
import scipy.signal

def fn_conv(input, params, hyper_params, backprop, dv_output=None):
    """
    Args:
        input: The input data to the layer function. [in_height] x [in_width] x [num_channels] x [batch_size] array
        params: Weight and bias information for the layer.
            params['W']: layer weights, [filter_height] x [filter_width] x [filter_depth] x [num_filters] array
            params['b']: layer bias, [num_filters] x 1 array
        hyper_params: Optional, could include information such as stride and padding.
        backprop: Boolean stating whether or not to compute the output terms for backpropagation.
        dv_output: The partial derivative of the loss with respect to each element in the output matrix. Only passed in when backprop is set to true. Same size as output.

    Returns:
        output: Output of layer, [out_height] x [out_width] x [num_filters] x [batch_size] array
        dv_input: The derivative of the loss with respect to the input. Same size as input.
        grad: The gradient term that you will use to update the weights defined in params and train your network. Dictionary with same structure as params.
            grad['W']: gradient wrt weights, same size as params['W']
            grad['b']: gradient wrt bias, same size as params['b']
    """

    in_height, in_width, num_channels, batch_size = input.shape
    filter_height, filter_width, filter_depth, num_filters = params['W'].shape
    out_height = in_height - filter_height + 1
    out_width = in_width - filter_width + 1

    assert filter_depth == num_channels, 'Filter depth does not match number of input channels'

    # Initialize
    output = np.zeros((out_height, out_width, num_filters, batch_size))
    dv_input = np.zeros(0)
    grad = {'W': np.zeros(0),
            'b': np.zeros(0)}
    
    # TODO: FORWARD CODE
    #       Update output with values
    

    for i in range(batch_size):
        data = input[:,:,:,i]
        for j in range(num_filters):
            filter = params['W'][:,:,:,j]
            bias = params['b'][j]
            # print(np.shape(output[:,:,j,i]))
            output[:,:,j,i]= scipy.signal.convolve(data,np.flip(filter),mode='valid',method = 'direct')[:,:,0]+np.ones((out_height,out_width))*bias

    if backprop:
        assert dv_output is not None
        dv_input = np.zeros(input.shape)
        grad['W'] = np.zeros(params['W'].shape)
        grad['b'] = np.zeros(params['b'].shape)
        
        # TODO: BACKPROP CODE
        #       Update dv_input and grad with values
        
        for j in range(num_filters):
            grad['b'][j] = np.sum(dv_output[:,:,j])/batch_size
                
        for i in range(batch_size):
            for j in range(num_filters):
                for k in range(filter_depth):
                    output_padded = np.zeros((in_height+filter_height-1,in_width + filter_width - 1))
                    output_padded[filter_height-1:-(filter_height-1),filter_width-1:-(filter_width-1)] = dv_output[:,:,j,i]
                    dv_input[:,:,k,i] +=  scipy.signal.convolve(output_padded,params['W'][:,:,k,j],mode = 'valid',method='direct')
    
        for i in range(batch_size):
            for j in range(num_filters):
                grad['W'][:,:,:,j] += scipy.signal.convolve(np.expand_dims(np.flip(dv_output[:,:,j,i]),axis = 2),input[:,:,:,i],mode = 'valid',method='direct')#flip!!
        grad['W']/=batch_size


    return output, dv_input, grad
