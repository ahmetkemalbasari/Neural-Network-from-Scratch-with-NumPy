import numpy as np
import time
from typing import List

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

    def __init__(self, input_size):
        self._z = 0
        self._a = 0
        self._inputs: np.ndarray = None
        self.weights = np.random.uniform(-0.1, 0.1, input_size)
    
    def forward(self, inputs):
        self._inputs = inputs
        self._z = np.dot(inputs, self.weights)
        self._a = ACTIVATIONS["sigmoid"]["func"](self._z)
        return self._a

    def backprop(self, grad, learning_rate):
        delta = grad * ACTIVATIONS["sigmoid"]["grad"](self._z, self._a)
        n_grads = self.weights * delta 

        self.weights -= delta * learning_rate
        return n_grads


class Layer:

    def __init__(self, neuron_size, prev_neuron_size):
        self.prev_neuron_size = prev_neuron_size
        self.neurons = [Neuron(prev_neuron_size) for i in range(neuron_size)]

    def forward(self, inputs):
        return np.stack([neuron.forward(inputs) for neuron in self.neurons])

    def backprop(self, grads, learning_rate):
        next_grad = np.zeros(self.prev_neuron_size, dtype = float)
        for neuron, grad in zip(self.neurons, grads):
            next_grad += neuron.backprop(grad, learning_rate)
        return next_grad

class General:

    def __init__(self, layer_sizes): # [12, 8, 4, 2] 12 -> direkt 12 input, 8 -> gerçekten 8 tane neuron
        self.layers: List[Layer] = []
        for i, layer_size in enumerate(layer_sizes):
            if i == 0:
                continue
            self.layers.append(Layer(layer_size, layer_sizes[i-1]))
    
    def forward(self, inputs: np.ndarray):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backprop(self, X: np.ndarray, Y: np.ndarray, learning_rate):
        pred = self.forward(X)
        error = pred - Y
        grads = 2 * error

        for layer in reversed(self.layers):
            grads = layer.backprop(grads)

        return error
    
