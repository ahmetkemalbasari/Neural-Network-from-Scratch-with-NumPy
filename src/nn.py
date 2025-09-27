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

    def __init__(self, input_size, activation):
        self._z = 0
        self._a = 0
        self._act = ACTIVATIONS[activation]
        self._inputs: np.ndarray = None
        self.bias = np.random.uniform(-0.1, 0.1)
        self.weights = np.random.uniform(-0.1, 0.1, input_size)
    
    def forward(self, inputs):
        self._inputs = inputs
        self._z = np.dot(self.weights, inputs) + self.bias
        self._a = self._act["func"](self._z)
        return self._a

    def backprop(self, grad, learning_rate):
        delta = grad * self._act["grad"](self._z, self._a)
        n_grads = self.weights * delta 

        self.bias -= learning_rate * delta
        self.weights -=  learning_rate  * delta * self._inputs
        return n_grads


class Layer:

    def __init__(self, neuron_size, prev_neuron_size, activation):
        self.prev_neuron_size = prev_neuron_size
        self.neurons = [Neuron(prev_neuron_size, activation) for i in range(neuron_size)]

    def forward(self, inputs):
        return np.stack([neuron.forward(inputs) for neuron in self.neurons])

    def backprop(self, grads, learning_rate):
        next_grad = np.zeros(self.prev_neuron_size, dtype = float)
        for neuron, grad in zip(self.neurons, grads):
            next_grad += neuron.backprop(grad, learning_rate)
        return next_grad

class General:

    def __init__(self, layer_sizes, activations): # [12, 8, 4, 2] 12 -> direkt 12 input, 8 -> gerçekten 8 tane neuron
        self.layers: List[Layer] = []
        self.layer_sizes = layer_sizes
        self.activations = activations

        for neuron_size, prev_neuron_size, act in zip(layer_sizes[1:], layer_sizes, activations):
            self.layers.append(Layer(neuron_size, prev_neuron_size, act))
    
    def forward(self, inputs: np.ndarray):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backprop(self, X: np.ndarray, Y: np.ndarray, learning_rate):
        pred = self.forward(X)
        error = pred - Y
        grads = 2 * error

        for layer in reversed(self.layers):
            grads = layer.backprop(grads, learning_rate)

        return np.mean(error**2)
    
    def train(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.01) -> float:
        nsamples = X.shape[0]

        idx = np.random.permutation(nsamples)
        Xs, Ys = X[idx], Y[idx]

        total_loss = 0.0
        for x, y in zip(Xs, Ys):
            total_loss += self.backprop(x, y, lr)
        return total_loss / nsamples
    
    def save(self, path):
        params = {
            "layer_sizes": np.array(self.layer_sizes),
            "activations": np.array(self.activations, dtype=object)
        }
        for l, layer in enumerate(self.layers):
            W = np.stack([n.weights for n in layer.neurons])
            b = np.stack([n.bias    for n in layer.neurons])
            params[f"W{l}"] = W
            params[f"b{l}"] = b

        with open(path, "wb") as f:
            np.savez(f, **params)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = np.load(f, allow_pickle=True)

            layer_sizes = data["layer_sizes"].tolist()
            activations = data["activations"].tolist()

            model = cls(layer_sizes=layer_sizes, activations=activations)
            for i, l in enumerate(model.layers):
                W = data[f"W{i}"]
                b = data[f"b{i}"]
                for neuron, w_vec, bias in zip(l.neurons, W, b):
                    neuron.weights = w_vec.copy()
                    neuron.bias    = float(bias)
            return model
