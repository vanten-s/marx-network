
import math
import random

def ReLU(x):
    return (abs(x)+x)/2

def sigmoid(x):
    return 1/(1+math.exp(-x))

def fitness(expected, got):
    return sum([(expected[i]-got[i])**2 for i in range(0, len(expected))])

def iterate()

class Node:
    def __init__(self, n_inputs, weights, bias, activation_func):
        self.n_inputs = n_inputs
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func

    def get_output(self, inputs):
        outputs = []
        for i in range(0, len(inputs)):
            outputs.append(self.weights[i]*inputs[i])

        return self.activation_func(sum([inputs[i] * self.weights[i] for i in range(0, len(inputs))])+self.bias)

class Layer:
    def __init__(self, n_nodes, n_inputs, weights, biases, activation_func):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.weights = weights
        self.biases = biases
        self.activation_func = activation_func
        self.nodes = [Node(n_inputs, weights[i], biases[i], activation_func) for i in range(0, n_nodes)]

    def randomise_factors(self, radius):
        for node in self.nodes:
            node.bias += (2*random.random()-1) * radius
            for i in range(0, len(node.weights)):
                node.weights[i] += (2*random.random()-1) * radius

    def get_fitness(self, inputs, expected):
        return fitness(expected, self.get_output(inputs))

    def get_output(self, inputs):
        return [node.get_output(inputs) for node in self.nodes]

class Network:
    def __init__(self, n_inputs, n_layers, n_nodes_per_layer, n_outputs, weights=None, biases=None, activation_functions):
        self.activation_functions = activation_functions
        self.n_inputs = n_inputs
        self.n_layers = n_layers


