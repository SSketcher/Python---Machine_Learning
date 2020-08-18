import numpy as np
import pandas as pd

class Regression(object):
    def __init__(self, unites = 1):
        """
        instantiate a new Linear Regression model

        :param unites: number of features
        """
        self.m = np.zeros(unites)
        self.b = 0
        self.log = []       #log contains valeu of cost function and m and b for each epoch

    def fit(self,X: np.array, Y: np.array, epochs = 1, rate = 0.001):
        """
        fit the Linear Regression model on the training data

        :param data: data to fit the model on       array(N x features)
        :param labels: labels of the training data      array(N x 1)
        :param epochs: number of training iterations 
        """
        for e in range(epochs):
            length = float(len(Y))
            acc = [Y]
            Ypred = self.predict(X)
            dm = (-2/length) * sum(X * (Y - Ypred))
            db = (-2/length) * sum(Y - Ypred)
            self.m = self.m - (rate * dm)
            self.b = self.b - (rate * db)
            acc.append(Ypred)
            print("------- Epoch " + str(e + 1) + " -------")
            print("Cost: ", self.__cost_func(acc))
            self.log.append((self.__cost_func(acc), (self.m, self.b)))

    def predict(self,X: np.array):
        """
        predict output useing the Linear Regression model

        :param X: inputs for the model      array(1 x features)

        :returns Y for given X value/s
        """
        return (self.m * X) + self.b

    def __cost_func(self, acc):     #calculates the cost function (mean squared error)
        length = len(acc)
        return (1/length) * sum(err**2 for err in (acc[0] - acc[1]))    #acc[0] = Y, acc[1] = Ypred


