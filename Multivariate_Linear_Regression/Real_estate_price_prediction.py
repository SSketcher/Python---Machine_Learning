from linear_regression import Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = pd.read_csv('Real estate.csv', header = 0)
data.pop('No')
headers = [str(col) for col in data.columns]
data = data.sample(frac=1)

X = data.iloc[:, :6].values
Y = data.iloc[:, 6:].values

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']

fig, axs = plt.subplots(2, 3)
fig.suptitle('Effect of a given parameter on real estate price')
number = 0
for i in range(2):
    for j in range(3):
        axs[i, j].scatter(X[:, number], Y, color = colors[number], marker = 'h')
        number += 1
number = 0
for ax in axs.flat:
    ax.set(xlabel = headers[number], ylabel = 'price per unit area')
    number += 1
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Price per unit area by the geographic coordinates')
ax.set_xlabel("Latitude")
ax.set_ylabel("Longitude")
ax.set_zlabel("Price per unit area")
ax.scatter(X[:, 4], X[:, 5], Y, marker='o', s=4, color='red', edgecolor='red')
plt.show()

X = X.reshape((len(X), 6, 1))
Y = Y.reshape((len(Y), 1))

print(X[0] * X[1])

model = Regression(unites = 6)
model.fit(X, Y, epochs = 10)