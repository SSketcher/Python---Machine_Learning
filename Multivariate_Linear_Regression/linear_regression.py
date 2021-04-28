import numpy as np

class Regression(object):
    def __init__(self, unites = 1):
        """
        instantiate a new Linear Regression model

        :param unites: number of features
        """
        self.unites = unites
        self.m = np.zeros(unites).reshape(unites, 1)
        print(self.m)
        self.b = 0
        self.log = []       #log contains valeu of cost function and m and b for each epoch

    def fit(self,X: np.array, Y: np.array, epochs = 1, batch = 10, rate = 0.001):
        """
        fit the Linear Regression model on the training data

        :param X: data to fit the model on       array(N x features x 1)
        :param Y: labels of the training data      array(N x 1 x 1)
        :param epochs: number of training iterations
        :param batch: size of batch for gradient descent iteration in ech epoch
        :param rate: learning rate of the model
        """
        for e in range(epochs):
            batchs = self.__batch(X, Y, batch)
            acc = [[],[]]
            for b in batchs:
                c = input("----------STOP----------")
                for elm in b:
                    x = elm[:self.unites-1]
                    y = elm[self.unites-1:]
                    print('X: ', x)
                    print('Y: ', y)
                    print('m: ', self.m)
                    ypred = self.predict(x)
                    print('Y predicted: ', ypred)
                    acc[0].append(y)
                    acc[1].append(ypred)
                self.__grad_decent(x, y, ypred, rate)       #Applying batch gradient descent
            print("------- Epoch " + str(e + 1) + " -------")
            print("Cost: ", self.__cost_func(acc))
            self.log.append((self.__cost_func(acc), (self.m, self.b)))

    def predict(self, x: np.array):
        """
        predict output useing the Linear Regression model

        :param X: inputs for the model      array(1 x features)

        :returns Y for given X value/s
        """
        return (sum(self.m * x) + self.b).astype(float)

    def __cost_func(self, acc):     #calculates the cost function (mean squared error)
        length = len(acc[0])
        return (1/(2 * length)) * sum(err**2 for err in (np.array(acc[1]) - np.array(acc[0])))    #acc[0] = Y, acc[1] = Ypred

    def __grad_decent(self, X, Y, Ypred, rate):
        length = len(Y)
        dJ = 0              #This is sum of a partial differentials of the cost function(times length) for each data point
        for i in range(length):
            dJ += (Ypred[i] - Y[i]) * X[i]
        self.b = self.b - ((rate/length) * sum(Ypred - Y))      #Gradien descent for b
        for i in range(self.unites):
            self.m[i] = self.m[i] - ((rate/length) * dJ)        #Gradien descent for each m

    def features_norm(self):
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
        if(len(xbatch) != 0):
            batches.append(np.array(xbatch))
        return batches

