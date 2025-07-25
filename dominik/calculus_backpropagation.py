
# source of the caclulations: https://www.youtube.com/watch?v=tIeHLnjs5U8

import numpy as np 

# aLayer - output of the layer
# y - desired output
def get_cost(aLayer, y):    
    return (aLayer - y) ** 2

# aPrevLayer - output of the previous layer 
def get_weighted_sum(weight, bias, aPrevLayer):
    return weight * aPrevLayer + bias

# aPrevLayer is the output of the previous layer 
def get_output_of_layer(weight, bias, aPrevLayer):
    # weighted sum 
    zL = get_weighted_sum(weight, bias, aPrevLayer) 
    return sigmoid(zL)

def sigmoid(x):
 return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

####
# Ratio functions for backpropagation
####

# aLayer - output of the layer
# y - desired output
def ratio_cost_nudge_to_output(aLayer, y):
    return 2 * (aLayer - y)

def ratio_output_to_weighted_sum(weight, bias, aPrevLayer):
    weighted_sum = get_weighted_sum(weight, bias, aPrevLayer)
    return sigmoid_derivative(weighted_sum)

# alfaC0 / alfaW(L)
def get_sensitivity_cost_to_prev_layer(weight, bias, aPrevLayer, y):
    aLayer = get_output_of_layer(weight, bias, aPrevLayer)  # Example weights and bias 

    return weight * \
        ratio_output_to_weighted_sum(weight, bias, aPrevLayer) * \
        ratio_cost_nudge_to_output(aLayer, y)

# shows how the weight should be updated for the next iteration
# weight - current weight
# bias - current bias
# aPrevLayer - output of the previous layer
# y - desired output

def update_weight(weight, bias, aPrevLayer, y, learning_rate):
    
    z = get_weighted_sum(weight, bias, aPrevLayer)
    a = sigmoid(z)

    # derivative of cost with respect to the output
    dcost_dw = 2 * (a - y) * sigmoid_derivative(z) * aPrevLayer

    # update the weight
    new_weight = weight - learning_rate * dcost_dw

    return new_weight



y = 1 # desired output

learning_rate = 0.1 # learning rate for weight update
weight_last = 0.9084 # weight of the last layer
bias_last = 0.1 # bias of the last layer

aPrevious = 0.50 # output from the previous layer
aLast = get_output_of_layer(weight_last, bias_last, aPrevious) # output from the last layer

sensitivity_cost_to_prev_layer = get_sensitivity_cost_to_prev_layer(weight_last, bias_last, aPrevious, y)

new_weight = update_weight(weight_last, bias_last, aPrevious, y, learning_rate)


print("Sensitivity cost to prev layer:", sensitivity_cost_to_prev_layer)

print("New weight after update:", new_weight)

