import numpy as np

class Regression(object):
    def __init__(self, unites = 1):
        """
        instantiate a new Linear Regression model

        :param unites: number of features
        """
        self.unites = unites
        self.m = np.zeros(unites)
        self.b = 0
        self.log = []       #log contains valeu of cost function and m and b for each epoch

    def fit(self,X: np.array, Y: np.array, epochs = 1, batch = 1, rate = 0.001):
        """
        fit the Linear Regression model on the training data

        :param X: data to fit the model on       array(N x features)
        :param Y: labels of the training data      array(N x 1)
        :param epochs: number of training iterations
        :param batch: size of batch for gradient descent iteration in ech epoch
        :param rate: learning rate of the model
        """
        
        for e in range(epochs):
            acc = []
            ##
            print("------- Epoch " + str(e + 1) + " -------")
            print("Cost: ", self.__cost_func(acc))
            self.log.append((self.__cost_func(acc), (self.m, self.b)))

    def predict(self,X: np.array):
        """
        predict output useing the Linear Regression model

        :param X: inputs for the model      array(1 x features)

        :returns Y for given X value/s
        """
        return ((self.m * X) + self.b).astype(float)

    def __cost_func(self, acc):     #calculates the cost function (mean squared error)
        length = len(acc[0])
        return (1/(2 * length)) * sum(err**2 for err in (acc[1] - acc[0]))    #acc[0] = Y, acc[1] = Ypred

    def __grad_decent(self, x, y, ypred, rate):
        self.b = self.b - (rate * (1/self.unites) * sum(ypred - y))
        for i in range(self.unites):
            self.m[i] = self.m[i] - (rate * (1/self.unites) * sum((ypred - y) * x[i]))


    def __batch(self, X: np.array, Y: np.array, batch):      #method dividing dataset in to baches 
        df = np.hstack((X, Y))
        np.random.shuffle(df)       #shuffleing the dataset befor dividing
        batches = []
        xbatch = []
        for element in df:
            xbatch.append(element)
            if(len(xbatch) == batch):
                batches.append(np.array(xbatch))
                xbatch = []
            elif(len(xbatch) != 0):
                batches.append(np.array(xbatch))
        return batches

