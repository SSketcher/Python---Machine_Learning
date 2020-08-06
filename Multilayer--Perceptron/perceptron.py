import numpy as np 
import pandas as pd

class Perceptron(object):
    def __init__(self, layers = [], activation = 'relu'):
        self.activation = activation
        self.layers = layers
        self.layers[0].weights = np.zeros(self.layers[0].size)
        prev_layer = self.layers[0]
        for layer in self.layers[1:]:
            layer.set_weights(prev_layer)
            prev_layer = layer

    def fit(self):
        pass

    def eval(self,X: np.array, Y: np.array):
        acc = []
        for x, y in zip(X, Y):
            yp = predict(x)
            acc.append((y, yp))
        return self.__accuracy(acc)

    def predict(self, X: np.array):
        for layer in self.layers[1:]:
            Xi = np.dot(X, layer.weights[:][1:]) - layer.weights[:][0]
            output = self.__activation_fnc(Xi)
        return  output

    def __gradient(self,X: np.array, Y: np.array, Yp: np.array):
        n = len(self.layers) - 2
        for i in range(n):
            deltJ = np.multiply(-(Y - Yp), self.__activation_fnc_div())
            dJdW = np.dot()

    def __activation_fnc(self,X):     #function applying selected activation function
        if self.activation == 'relu':
            return self.__relu(X)
        elif self.activation == 'sigmoid':
            return self.__sigmoid(X)
        elif self.activation == 'heaviside':
            return self.__heaviside(X)

    def __activation_fnc_div(self,X):     #function applying selected activation function
        if self.activation == 'relu':
            return self.__relu_div(X)
        elif self.activation == 'sigmoid':
            return self.__sigmoid_div(X)
        elif self.activation == 'heaviside':
            return self.__heaviside_div(X)

    def __relu(self, X: np.array):     #rectifier activation function
        return np.log((1 + np.exp(X)), np.e)

    def __relu_div(self, X: np.array):
        return 1 / (1 + np.exp(-X))

    def __sigmoid(self, X: np.array):       #sigmoid activation function
        return 1 / (1 + np.exp(-X))

    def __sigmoid_div(self, X: np.array):
        return np.exp(X) / (1 + np.exp(-X))

    def __heaviside(self, X: np.array):     #heaviside activation function
        return np.where(X >= 0, 1, 0)

    def __heaviside_div(self, X: np.array):
        return np.where(X == 0, 1, 0)

    def __randomize(self, data: np.array, labels: np.array):        #function shuffling dataset
        df = np.hstack((data, labels))
        np.random.shuffle(df)
        df = np.hsplit(df, np.array([len(data[0]), len(data)]))
        return (df[0], df[1])

    def __cost_func(self, acc: list()):       #calculating valeu of cost function
        y = np.array(acc[:][0])
        yp = np.array(acc[:][1])
        return sum(0.5 * (y - yp))

    def __accuracy(self,acc: list()):       #calculating accuracy of predictions
        total_population = len(acc)
        true_positive = 0
        true_negative = 0
        for row in acc:
            a,b = row
            if a == np.round(b) and a == 0:
                true_negative += 1
            if a == np.round(b) and a == 1:
                true_positive += 1
        accuracy = (true_positive + true_negative) / total_population
        return accuracy


class Layer(object):
    def __init__(self,size = 1):
        self.size = size
        self.weights = None

    def set_weights(self, prev_layer):
        self.weights = np.zeros((self.size, prev_layer.size + 1))