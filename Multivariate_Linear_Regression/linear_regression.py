import numpy as np

class Regression(object):
    def __init__(self, unites = 1):
        """
        instantiate a new Linear Regression model

        :param unites: number of features
        """
        self.unites = unites
        self.m = np.zeros(unites).reshape(unites, 1)
        print('m: ', self.m)
        self.b = 0
        self.log = []       #log contains valeu of cost function and m and b for each epoch

    def fit(self,X: np.array, Y: np.array, epochs = 1, batch_size = 10, rate = 0.001):
        """
        fit the Linear Regression model on the training data

        :param X: data to fit the model on       array(N x features x 1)
        :param Y: labels of the training data      array(N x 1 x 1)
        :param epochs: number of training iterations
        :param batch: size of batch for gradient descent iteration in ech epoch
        :param rate: learning rate of the model
        """
        for e in range(epochs):
            batchs = self.__batch(X, Y, batch_size)     #Shuffling training set and dividing it into batches
            acc = [[],[]]
            for batch in batchs:
                Xb = batch[:, :self.unites]
                Yb = batch[:, self.unites:]
                Y_pred = []                
                for i in range(len(batch)):
                    y_pred = self.predict(Xb[i])
                    print(y_pred)
                    Y_pred.append(y_pred)
                acc[0].extend(Yb)
                acc[1].extend(np.array(Y_pred))
                _ = input('STOP')
                self.__grad_decent(Xb, Yb, Y_pred, rate)       #Applying batch gradient descent
            print("------- Epoch " + str(e + 1) + " -------")
            print("Cost: ", self.__cost_func(acc))
            self.log.append((self.__cost_func(acc), (self.m, self.b)))

    def predict(self, x: np.array):
        """
        predict output useing the Linear Regression model

        :param X: inputs for the model      array(features x 1)

        :returns Y for given X value/s
        """
        return (sum(self.m * x) + self.b).astype(float)

    def features_norm(self, X: np.array):
        """
        performing mean normalization on given dataset X

        :param X: inputs for the model      array(N x features x 1)

        :returns X_norm: a normalized version of X      array(N x features x 1)
        :returns mu: the mean value      array(features x 1)
        :returns sigma: the standard deviation      array(features x 1)
        """
        mu = np.mean(X, axis = 0) 
        sigma = np.std(X, axis= 0, ddof = 1)        # Standard deviation
        X_norm = (X - mu)/sigma
        return X_norm, mu, sigma

    def features_scal(self, X: np.array):
        """
        performing features scaling on given dataset X

        :param X: inputs for the model      array(N x features x 1)

        :returns X_scal: a scaled version of X      array(N x features x 1)
        :returns sigma: the standard deviation      array(features x 1)
        """
        sigma = np.std(X, axis= 0, ddof = 1)        # Standard deviation
        X_scal = X/sigma
        return X_scal, sigma


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
        if(len(xbatch) != 0):
            batches.append(np.array(xbatch))
        return batches

    def __cost_func(self, acc):     #calculates the cost function (mean squared error)
        length = len(acc[0])
        return (1/(2 * length)) * sum(err**2 for err in np.subtract(np.array(acc[1]), np.array(acc[0])))    #acc[0] = Y, acc[1] = Ypred

    def __grad_decent(self, X, Y, Y_pred, rate):
        length = len(Y)
        dJ = [0, 0]              #This is sum of a partial differentials of the cost function(times length) for each data point
        for i in range(length):
            dJ[0] += (Y_pred[i] - Y[i])         #dJ has shape of 1 x 1
            dJ[1] += (Y_pred[i] - Y[i]) * X[i]          #dJ has shape of features x 1
        print(dJ[0])
        print(dJ[1])
        self.b = np.subtract(self.b, ((rate/length) * dJ[0]))      #Gradien descent for b
        self.m = np.subtract(self.m, ((rate/length) * dJ[1]))        #Gradien descent for each m
