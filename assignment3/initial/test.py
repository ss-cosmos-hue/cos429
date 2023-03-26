import sys
sys.path += ['layers']
import numpy as np
from loss_crossentropy import loss_crossentropy

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
def test(model,input,true_label):
    output,layer_acts = inference(model,input)
    numinput = input.shape[-1]

    loss, _ = loss_crossentropy(output, true_label, None, backprop = False)
    #not sure whether backprop should be T or F
    pred = np.argmax(output,axis = 0)
    accuracy = np.count_nonzero(pred==true_label)/numinput
    return accuracy,loss