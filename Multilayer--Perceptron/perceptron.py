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

    def fit(self, X: np.array, Y:np.array, epochs = 1, rate = 0.01):
        n = len(self.layers) - 2
        for e in range(epochs):
            for x, y in zip(X, Y):
                Z = []
                A = []
                W = []
                acc = []
                A.append(x)
                dJdW = []
                for layer in self.layers[1:]:
                    zi = np.dot(layer.weights, x) + layer.bias
                    ai = self.__activation_fnc(zi)
                    W.append(layer.weights)
                    Z.append(zi)
                    A.append(ai)
                    x = zi

                W.reverse()
                A.reverse()
                Z.reverse()
                acc.append((y, A[0]))

                dN = np.multiply(-(y - A[0]), self.__activation_fnc_div(Z[0]))
                print(dN)
                dJdW.append(np.dot(A[0].T, dN))
                print(dJdW[0])

                for i in range(n):
                    print(i)
                    print(dJdW[i])
                    print(W[i])
                    print("----------------------")
                    dN = np.dot(dJdW[i], W[i]) * self.__activation_fnc_div(Z[1 + i])
                    print(dN)
                    print(A[1 + i])
                    dJdW.append(np.dot(A[1 + i].T, dN.T))
                
                for i in range(n + 1):
                    self.layers[1 + i].weights -= rate * dJdW[i]

            print("------- Epoch " + str(e) + " -------")
            print("Accuracy: ", self.__accuracy(acc))
            print("Cost: ", self.__cost_func(acc))
                

    def eval(self,X: np.array, Y: np.array):
        acc = []
        for x, y in zip(X, Y):
            yp = self.predict(x)
            acc.append((y, yp))
        return self.__accuracy(acc)

    def predict(self, X: np.array):
        for layer in self.layers[1:]:
            Xi = np.dot(layer.weights, X) + layer.bias
            output = self.__activation_fnc(Xi)
            X = output
        return  output

    def __activation_fnc(self, X: np.array):     #function applying selected activation function
        if self.activation == 'relu':
            return np.log((1 + np.exp(X)), np.e)        #rectifier activation function
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))       #sigmoid activation function
        elif self.activation == 'heaviside':
            return np.where(X >= 0, 1, 0)     #heaviside activation function

    def __activation_fnc_div(self,X):     #function applying derivative of selected activation function
        if self.activation == 'relu':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'sigmoid':
            return np.exp(X) / (1 + np.exp(-X))
        elif self.activation == 'heaviside':
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
        self.bias = np.zeros(self.size)


    def set_weights(self, prev_layer):
        self.weights = np.random.rand(self.size, prev_layer.size)