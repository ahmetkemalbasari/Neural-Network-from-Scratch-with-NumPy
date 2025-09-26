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
    
    def forward(self, inputs):
        arr0 = np.array(inputs)
        arr1 = np.array(self.weights)
        self.input = np.dot(arr0, arr1)
        self.output = ACTIVATIONS["sigmoid"]["func"](self.input)
        return self.output
    

class Layer:

    def __init__(self, neuronSize, prevNeuronSize):
        self.neurons = [Neuron(prevNeuronSize) for i in range(neuronSize)]

    def forward(self, inputs):
        outputs = list()
        [outputs.append(neuron.forward(inputs)) for neuron in self.neurons]
        return outputs
            


class General:

    def __init__(self, layerSizes, inputs): # [12, 8, 4, 2] 12 -> direkt 12 input, 8 -> gerçekten 8 tane neuron
        self.layers = list()
        self.inputs = inputs
        for i, layerSize in enumerate(layerSizes):
            if i == 0:
                continue
            self.layers.append(Layer(layerSize, layerSizes[i-1]))
    
    def forward(self):
        outputs = self.inputs
        for i, layer in enumerate(self.layers):
            if i == 0:
                #layer.forward(self.inputs)
                outputs = layer.forward(outputs)
                continue
            #prev_outputs = [neuron.output for neuron in self.layers[i - 1].neurons]
            #layer.forward(prev_outputs)
            outputs = layer.forward(outputs)
        #[print(neuron.output) for neuron in self.layers[len(self.layers)-1].neurons]
        print(outputs)
