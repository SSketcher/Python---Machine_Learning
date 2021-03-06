from linear_regression import Regression
import numpy as np
import matplotlib.pyplot as plt 


#Creating dataset
eps = np.random.randint(5, size=50)/2
x = np.linspace(1.0, 50.0, num = 50)
y = 3/4 * x + 2

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
Y = np.array([3.5, 2.0, 5.0 , 2.5, 6.0 , 7.2, 5.4 , 8, 7.5, 11.2, 10.5, 13.4, 14.0, 13.0, 16.5])


#Ploting dataset
plt.plot(X, Y, 'gh')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear distribution with random nois')
plt.grid(True)
plt.show()


#Performing linear regression
model = Regression()      #Declaration of the model
model.fit(X, Y, epochs = 50)        #Training the model

Yp = model.predict(X)        #Testing the model


#Ploting dataset with change of regresion line
plt.plot(X, Y, 'gh')
for epoch in model.log:
    line = epoch[1][0] * X + epoch[1][1]
    plt.plot(X, line, color='lightcoral', linewidth=0.2, linestyle=':')
plt.plot(X, Yp, color='black', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dataset with change of regresion line')
plt.grid(True)
plt.show()


#Plotting change of cost function value by the epochs
x = np.linspace(1.0, 50.0, num = 50)
cost = [element[0] for element in model.log]

plt.plot(x, cost)
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.title('Cost function for each epoch')
plt.show()