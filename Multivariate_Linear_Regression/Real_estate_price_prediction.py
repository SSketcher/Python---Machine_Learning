from linear_regression import Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#Loading data
data = pd.read_csv('resources/Real_estate.csv', header = 0)
data.pop('No')
headers = [str(col) for col in data.columns]
data = data.sample(frac=1)

X = data.iloc[:, :6].to_numpy()
Y = data.iloc[:, 6:].to_numpy() 


#Impact of each parameter on the real estate price
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']

fig, axs = plt.subplots(2, 3)
fig.suptitle('Impact of each parameter on the real estate price')
number = 0
for i in range(2):
    for j in range(3):
        axs[i, j].scatter(X[:, number], Y, color = colors[number], marker = 'h', s = 3.5)
        number += 1
number = 0
for ax in axs.flat:
    ax.set(xlabel = headers[number], ylabel = 'Price per unit area')
    number += 1
plt.show()


#3D plot showing price per unit area by the geographic coordinates
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Price per unit area by the geographic coordinates')
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Price per unit area")
ax.scatter(X[:, 4], X[:, 5], Y, marker='o', s=4, color='red', edgecolor='red')
plt.show()


#Performing linear regression
model = Regression(unites = 6)      #Declaration of the model
X, mu, sugma = model.features_norm(X)       #Data normalization
model.fit(X, Y, epochs = 100, batch_size = 105, rate = 0.05)        #Training the model


#Plotting change of cost function value by the epochs
x = np.linspace(1.0, 100.0, num = 100)
cost = [element[0] for element in model.log]

plt.plot(x, cost)
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.title('Cost function for each epoch')
plt.show()


#3D plot with linear regresion as a plane
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_title('Price per unit area by the geographic coordinates')
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Price per unit area")

m = model.m
b = model.b

x_min = np.nanmin(X[:, 4])
x_max = np.nanmax(X[:, 4])

y_min = np.nanmin(X[:, 5])
y_max = np.nanmax(X[:, 5])

x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)

Xs,Ys = np.meshgrid(x,y)
Zs = (b + m[4]*Xs + m[5]*Ys) #/ c

ax.scatter(X[:, 4], X[:, 5], Y, marker='o', s=4, color='red', edgecolor='red')
ax.plot_surface(Xs, Ys, Zs, alpha=0.45)

plt.legend(loc='upper left')
plt.show()