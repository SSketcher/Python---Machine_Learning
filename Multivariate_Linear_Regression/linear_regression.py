import numpy as np

class Regression(object):
    def __init__(self, unites = 1):
        """
        instantiate a new Linear Regression model

        :param unites: number of features
        """
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
            Xbatches = self.__batch(X, Y, batch)
            Yp = []
            Yg = []
            for bach in Xbatches:
                for i in range(len(batch)):
                    length = len(bach)
                    x = bach[:, :len(X[0])]
                    x = x.reshape((length, len(X[0])))
                    y = bach[:, -1]
                    y = y.reshape((length, 1))
                    yhat = self.predict(x)
                    dm = (-2.0/float(length)) * sum(np.multiply(x, (y - yhat))).reshape((len(X[0]), 1))
                    print(dm)
                    db = (-2.0/float(length)) * sum(y - yhat)
                    self.m = self.m - (rate * dm)
                    self.b = self.b - (rate * db)
                    print('yhat: ')
                    print(yhat)
                    Yg.extend(float(v) for v in y)
                    Yp.extend(float(v) for v in yhat)
            acc = [np.array(Yg)]
            acc.append(np.array(Yp))
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
        return (1/length) * sum(err**2 for err in (acc[0] - acc[1]))    #acc[0] = Y, acc[1] = Ypred

    def __grad_decent(self):
        pass

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

