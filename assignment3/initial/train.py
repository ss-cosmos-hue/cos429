import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = True

# You can modify the imports of this section to indicate
# whether to use the provided pyc or your own code for each of the four functions.
if use_pcode:
    # import the provided pyc implementation
    sys.path += ['pyc_code']
    from inference_ import inference
    from calc_gradient_ import calc_gradient
    from update_weights_ import update_weights
else:
    # import your own implementation
    from inference import inference
    from calc_gradient import calc_gradient
    from update_weights import update_weights
######################################################

def train(model, input, label, params, numIters):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        input: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            Free to add more parameters to this dictionary for your convenience of training.
        numIters: Number of training iterations
    '''
    # Initialize training parameters
    # Learning rate
    lr = params.get("learning_rate", .01)
    # Weight decay
    wd = params.get("weight_decay", .0005)
    # Batch size
    batch_size = params.get("batch_size", 128)
    # There is a good chance you will want to save your network model during/after
    # training. It is up to you where you save and how often you choose to back up
    # your model. By default the code saves the model in 'model.npz'.
    save_file = params.get("save_file", 'model.npz')

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd }

    num_inputs = input.shape[-1]
    loss = np.zeros((numIters,))

    for i in range(numIters):
        # TODO: One training iteration
        # Steps:
        #   (1) Select a subset of the input to use as a batch
        #   (2) Run inference on the batch
        #   (3) Calculate loss and determine accuracy
        #   (4) Calculate gradients
        #   (5) Update the weights of the model
        # Optionally,
        #   (1) Monitor the progress of training
        #   (2) Save your learnt model, using ``np.savez(save_file, **model)``
        
        #step1
        #How do we select a subset of the input? Can we just randomly select it?
        indices_of_batch =  np.random.choice(range(num_inputs),size = batch_size,replace=False)#randomly choose batch_size num of indices from [0,...,input-1]
        batch = input[:,indices_of_batch]
        mean = np.mean(batch,axis = 0)
        
        #todo normalization
        batch_label = label[indices_of_batch]#do we need to normalize label?
        #batch normalization
        
        #step2
        output,layer_acts = inference(model,batch)
        #output is expected to be a matrix, each column corresponding to an instance, whose probability of belonging to each class contained in each row
        
        #step3
        
        loss, dv_output = loss_crossentropy(output, batch_label, update_params, backprop = True)
        #not sure whether backprop should be T or F
        pred = np.argmax(output,axis = 0)
        accuracy = np.count_nonzero(pred==batch_label)/batch_size
        if i%50 == 0: 
            print("iter ",i, "accuracy ",accuracy,"loss ",loss)
        #step4
        grads = calc_gradient(model, input, layer_acts, dv_output)
        #step5
        model = update_weights(model,grads,update_params)
        
    #option2  
    np.savez(save_file, **model)
    return model, loss