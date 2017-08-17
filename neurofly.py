# -*- coding: utf-8 -*-
"""
Created on Fri Aug 04 21:14:13 2017

Neural fly

@author: Zan
"""


class Neuron:        
    def __init__(self):
        self.signal = []
        self.weight = []
        self.bias = 0.0
        self.threshold = 0.0
        self.output = 0.0
        return
        
    def set_weights(self, weights):
        self.weight = weights[:]
        return

    def set_threshold(self, threshold):
        self.threshold = threshold
        return
        
    def set_bias(self, bias):
        self.bias = bias
        
    def set_signal(self, signal):
        self.signal = signal[:]
        return
        
    def calculate(self):
        length = min(len(self.signal), len(self.weights))
        
        level = self.bias
        for i in range(length):
            level += self.weight[i] * self.signal[i]
        
        if level >= self.threshold:
            self.output = 1.0
        else:
            self.output = 0.0
            
        return self.output

    def get_output(self):
        return self.output



class Brain:
    def __init__(self, levels):
        """ Levels is an array containing size of each level """
        self.levels = levels
        if self.levels < 1:
            return False
        
        # Create neuron map
        self.neurons = []
        for level in self.levels:
            self.neurons.append([])
            
            for i in range(level):
                self.neurons[-1].append(Neuron())
        
        self.connections = []
        return True
        
    def set_connections(self, connections):
        """ Connections is an 3-dimensional array of connection indices """
        self.connections = connections[:]
        
        # Verify data
        if len(self.connections) != self.levels - 1:
            return False
            
        for (n,connection) in enumerate(self.connections):
            if len(connection) != len(self.neurons[n]):
                return False
        
        return True

    def set_biases(self, bias):
        self.biases = bias[:]
        
        for level in bias:
            
        
        return

    def set_inputs(self, inputs):
        self.inputs = inputs[:]

        if len(self.inputs) != len(self.neurons[0]):
            return False

        return True
        
    def get_outputs(self):
        self.outputs = []
        
        for neuron in self.neurons[-1]:
            self.outputs.append(neuron.get_output())
        
        return self.outputs[:]
        
    def calculate(self):
        # External inputs to first layer
        signal = []
        length = len(self.neurons[0])
        for i in range(length):
            signal.append([self.inputs[i]])
            
        # Then calculate layer by layer propagating signals downwards
        for i in range(self.levels):
            output = []
            length = len(self.neurons[i])
            for n in range(length):
                self.neurons[i][n].set_signal(signal[i])
                output.append(self.neurons[i][n].calculate())

            # Calculate inputs for all levels up to last one :)
            if i < self.levels - 1:
                # Initialize input signal array for next level neurons
                next_length = len(self.neurons[i+1])
                signal = [[] for x in range(next_length)]
                
                # Propagate outputs
                for n in range(length):
                    for x in self.connections[n]:
                        signal[x][n] = output[n]
            else:
                # Return output from last neuron level
                return output[:]
                
                

# Main

# Define brain by connections to other neurons
neural_net = [ [[0, 1, 2, 3], [0, 1, 2, 3]], \
               [[0], [0, 1], [0, 1], [1]], \
               [[0], [0]] \
                ]

levels = len(neural_net)

# Initialize biases randomly
bias = []
for level in range(levels) - 1:
    bias.append([])
    for neuron in level:
        bias.append()



brain = Brain(levels)
brain.set_connections(neural_net)