from linear_regression import Regression
import numpy as np
import matplotlib.pyplot as plt 


X = np.random.randint(20, size=30)
eps = np.random.randint(5, size=30)/2
Y = 3/4 * X + eps


plt.plot(X, Y, 'gh')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear distribution with random nois')
plt.grid(True)
plt.show()


model = Regression(unites = 1)
model.fit(X, Y, epochs = 50)


Yp = model.predict(X)

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



x = np.linspace(1.0, 50.0, num = 50)
cost = [element[0] for element in model.log]

plt.plot(x, cost, "bd")
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.title('Cost function for each epoch')
plt.show()