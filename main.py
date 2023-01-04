
import math
import random

def ReLU(x):
    return (abs(x)+x)/2

def sigmoid(x):
    return 1/(1+math.exp(-x))

def fitness(expected, got):
    return sum([(expected[i]-got[i])**2 for i in range(0, len(expected))])

def iterate(network, n_networks, radius, f, values):
    copies = [Network(network.n_inputs, network.n_layers, network.n_nodes_per_layer, network.n_outputs, network.activation_functions, network.weights, network.biases) for i in range(0, n_networks)]
    for copy in copies:
        copies.randomise_factors(radius)

    expectations = []
    for value in values:
        expectations.append(f(value))

    best = -1000
    best_network = copies[0]
    for copy in copies:
        fitness = copy.get_fitness(values, expectations)
        if fitness > best:
            best = fitness
            best_network = copy

    return best_network

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
    def __init__(self, n_inputs, n_layers, n_nodes_per_layer, n_outputs, activation_functions, weights=None, biases=None):
        self.activation_functions = activation_functions
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.nodes_per_layes = n_nodes_per_layer
        self.n_outputs = n_outputs
        if weights == None:
            weights = [[[1 for k in range(0, n_nodes_per_layer[i-1])] for j in range(0, n_nodes_per_layer[i])] for i in range(1, n_layers)]

        if biases == None:
            biases = [[1 for j in range(0, n_nodes_per_layer[i])] for i in range(1, n_layers)]

        self.weights = weights
        self.biases = biases

        self.layers = []
        for i in range(1, n_layers):
            self.layers.append(Layer(n_nodes_per_layer[i], n_nodes_per_layer[i-1], weights[i], biases[i], activation_functions[i]))

    def randomise_factors(self, radius):
        for layer in self.layers:
            layer.randomise_factors(radius)

    def get_output(self, inputs):
        prev_layer = inputs
        for layer in self.layers:
            prev_layer = layer.get_output(prev_layer)

        return prev_layer

    def get_fitness(self, inputs, expected):
        tot = 0
        output = self.get_output(inputs)
        for i in range(0, len(output)):
            tot += (output[i] - expected[i])**2

        return tot





