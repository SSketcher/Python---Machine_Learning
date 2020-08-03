import numpy as np
import pandas as pd

class Perceptron(object):
    def __init__(self, input_size = 1, activation = 'heaviside', rate = 0.01):
        """
        instantiate a new Perceptron

        :param input_size: number of features
        :param activation: used activation function sigmoid or heaviside
        :param rate: coefficient used to tune the model response to training data
        """
        self.weights = np.zeros(1 + input_size)
        self.activation = activation
        self.rate = rate

    def fit(self,data: np.array, labels: np.array, epochs = 1):
        """
        fit the Perceptron model on the training data

        :param data: data to fit the model on       array(N x features)
        :param labels: labels of the training data      array(N x 1)
        :param epochs: number of training iterations 
        """
        for i in range(epochs):
            data, labels = self.__randomize(data, labels)
            acc = []
            print('Epoch '+ str(i + 1) +' ----- ')
            for xi, yi in zip(data, labels):
                yp = self.predict(xi)
                dW = self.rate * (yi - yp)
                self.weights[1:] += dW * xi
                self.weights[0] += dW
                acc.append((yi, yp))
            print('Accuracy: ', self.__accuracy(acc))

    def eval(self, data: np.array, labels: np.array):
        """
        evaluate the Perceptron model on the test data

        :param data: data to evaluate the model     array(N x features)
        :param labels: labels of the test data      array(N x 1)
        """
        acc = []
        for xi, yi in zip(data, labels):
            yp = self.predict(xi)
            acc.append((yi, yp))
        print('Accuracy of evaluation:')
        print(str(self.__accuracy(acc) * 100) + '%')

    def predict(self,X: np.array):
        """
        predict output useing the Perceptron model

        :param X: inputs for the model      array(1 x features)

        :returns 1 or 0
        """
        output = np.dot(X, self.weights[1:]) - self.weights[0]
        return self.__activation_fnc(output)

    def __activation_fnc(self,X: np.array):     #function applying selected activation function
        if self.activation == 'heaviside':
            return self.__heaviside(X)
        elif self.activation == 'sigmoid':
            return self.__sigmoid(X)

    def __sigmoid(self, X: np.array):       #sigmoid activation function
        return 1 / (1 + np.exp(-X))

    def __heaviside(self, X: np.array):     #heaviside activation function
        return np.where(X >= 0, 1, 0)

    def __randomize(self, data: np.array, labels: np.array):        #function shuffling dataset
        df = np.hstack((data, labels))
        np.random.shuffle(df)
        df = np.hsplit(df, np.array([len(data[0]), len(data)]))
        return (df[0], df[1])

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