import layer_functions as lf
import numpy as np
import math


class Conv_Layer:
    def __init__(self, kernels):
        self.name = "Convolutional Layer"
        self.kernels = kernels

    def calculate_input(self, input):
        return lf.conv_op_3d(input, self.kernels)


class Relu_Layer:
    def __init__(self):
        self.name = "Relu Layer"

    def calculate_input(self, input):
        return lf.relu_3d(input)


class MaxPool_Layer:
    def __init__(self, size):
        self.name = "MaxPool Layer"
        self.size = size

    def calculate_input(self, input):
        return lf.maxPool(input, self.size)


class Normalize_Layer:
    def __init__(self):
        self.name = "Normalize Layer"

    def calculate_input(self, input):
        return lf.normalize_3d(input)


class Fully_Connected_Layer:
    def __init__(self, size):
        self.name = "Fully Connected Layer"
        self.size = size
        self.weights = None
        # Create a weight matrix using He initialization

    def calculate_input(self, input):
        self.weights = np.random.rand(input.size, self.size) \
                       * math.sqrt(2 / self.size)
        return lf.fullyconnected(input, self.weights)


class Softmax_Layer:
    def __init__(self):
        self.name = "Softmax Layer"

    def calculate_input(self, input):
        return lf.softmax(input)


class NN_Model:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        i = np.array([input])
        inp = np.moveaxis(i, 0, 2)
        for layer in self.layers:
            inp = layer.calculate_input(inp)
        return np.argmax(np.array(inp))

    def predicts(self, input):
        return list(map(self.predict, input))

    def accuracy(self,predicted_Y, true_Y):
        acclist = (np.array(predicted_Y) == np.array(true_Y))
        unique, counts = np.unique(acclist, return_counts=True)
        dict(zip(unique, counts))
        return counts[1]/acclist.size



    def train(self, train_X, train_Y):




        return 0

