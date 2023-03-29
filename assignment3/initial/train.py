import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy
from update_avegrads import update_avegrads
from test import test

######################################################
# Set use_pcode to True to use the provided pyc code
# for inference, calc_gradient, loss_crossentropy and update_weights
use_pcode = False

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

def train(model, X_train, y_train, X_test, y_test, params, numIters = 1000, rho = 0.99):
    '''
    This training function is written specifically for classification,
    since it uses crossentropy loss and tests accuracy assuming the final output
    layer is a softmax layer. These can be changed for more general use.
    Args:
        model: Dictionary holding the model
        X_train: [any dimensions] x [num_inputs]
        label: [num_inputs]
        params: Paramters for configuring training
            params["learning_rate"]
            params["weight_decay"]
            params["batch_size"]
            params["save_file"]
            params["print_step"]
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

    print_step = params.get("print_step", 10)

    test_step = params.get("test_step", 1)

    # update_params will be passed to your update_weights function.
    # This allows flexibility in case you want to implement extra features like momentum.
    update_params = {"learning_rate": lr,
                     "weight_decay": wd }

    num_inputs = X_train.shape[-1]
    train_accuracy = np.zeros((numIters,))
    train_loss = np.zeros((numIters,))
    test_accuracy = np.zeros((numIters,))
    test_loss = np.zeros((numIters,))
    
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
        
        # Step 1
        indices_of_batch = np.random.choice(range(num_inputs),size = batch_size,replace=False) # Randomly choose batch_size num of indices from [0,...,input-1]
        batch = X_train[...,indices_of_batch]
        batch_label = y_train[indices_of_batch]
        
        # Step 2
        output,layer_acts = inference(model,batch)

        # Step 3
        train_loss[i], dv_output = loss_crossentropy(output, batch_label, update_params, backprop = True)
        pred = np.argmax(output,axis = 0)
        train_accuracy[i] = np.count_nonzero(pred==batch_label) / batch_size

        # Optional 1
        if i % print_step == 0: 
            print("Iteration: ", i, "\tTrain Accuracy: ", train_accuracy[i], "\tTrain Loss: ", train_loss[i])
        if i % test_step == 0:
            test_accuracy[i], test_loss[i] = test(model, X_test, y_test)
            print("Test Accuracy: ", test_accuracy[i], "\tTest Loss: ", test_loss[i])
        # Step 4
        grads = calc_gradient(model, batch, layer_acts, dv_output)
        
        # Step 5
        if i == 0:
            momentum = grads
        else:
            momentum = update_avegrads(model, grads, momentum, rho=rho)
        model = update_weights(model, momentum, update_params)
        
    # Optional 2
    np.savez(save_file, **model)

    return model, train_accuracy, train_loss