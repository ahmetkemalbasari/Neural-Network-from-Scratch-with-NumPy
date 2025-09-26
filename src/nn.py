import numpy as np
import time

np.random.seed(int(time.time()))

ACTIVATIONS = {
    "sigmoid": {
        "func": lambda x: 1.0 / (1.0 + np.exp(-x)),
        "grad": lambda z, a: a * (1.0 - a)
    },
    "tanh": {
        "func": lambda x: np.tanh(x),
        "grad": lambda z, a: 1.0 - a*a
    },
    "relu": {
        "func": lambda x: np.maximum(0, x),
        "grad": lambda z, a: (z > 0).astype(float)
    },
    "leakyrelu": {
        "func": lambda x, alpha=0.01: np.where(x > 0, x, alpha * x),
        "grad": lambda z, a, alpha=0.01: np.where(z > 0, 1.0, alpha)
    },
    "linear": {
        "func": lambda x: x,
        "grad": lambda z, a: np.ones_like(z)
    },
}

class Neuron:

    def __init__(self, inputSize):
        self.input = 0
        self.output = 0
        self.weights = np.random.uniform(-0.1, 0.1, inputSize)
    
    def forward(self):
        return
        

class Layer:

    def __init__(self, neuronSize, prevNeuronSize):
        self.neurons = [Neuron(prevNeuronSize) for i in range(neuronSize)]


    def forward(self, inputs):
        for neuron in self.neurons:
            sum = 0
            for i, input in enumerate(inputs):
                sum += input * neuron.weights[i]
            neuron.output = sum


            


class General:

    def __init__(self, layerSizes, inputs): # [12, 8, 4, 2] 12 -> direkt 12 input, 8 -> gerçekten 8 tane neuron
        self.layers = list()
        for i, layerSize in enumerate(layerSizes):
            if i == 0:
                continue
            self.layers(Layer(layerSize, layerSizes[i-1]))
    
    def forward(self):
        for i, layer in enumerate(self.layers):
            layer.forward()

            layer.forward()