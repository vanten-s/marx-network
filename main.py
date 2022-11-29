
import math

def ReLU(x):
    return (math.abs(x)+x)/2

def sigmoid(x):
    return 1/(1+math.exp(-x))

class Node:
    def __init__(self, n_inputs, weights, bias, activation_func):
        self.n_inputs = n_inputs
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def get_output(inputs):
        outputs = []
        for i in range(0, len(inputs)):
            outputs[i] = self.weights[i]*inputs[i]

        return self.activation_func(sum([inputs[i] * self.weights[i] for i in range(0, len(inputs))])+bias)






