from linear_regression import Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = pd.read_csv('resources/Real estate.csv', header = 0)
data.pop('No')
headers = [str(col) for col in data.columns]
data = data.sample(frac=1)

X = data.iloc[:, :6].to_numpy()
Y = data.iloc[:, 6:].to_numpy() 

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']

fig, axs = plt.subplots(2, 3)
fig.suptitle('Effect of a given parameter on real estate price')
number = 0
for i in range(2):
    for j in range(3):
        axs[i, j].scatter(X[:, number], Y, color = colors[number], marker = 'h', s = 3.5)
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

model = Regression(unites = 6)
X, mu, sugma = model.features_norm(X)
model.fit(X, Y, epochs = 100, batch_size = 105, rate = 0.05)

x = np.linspace(1.0, 100.0, num = 100)
cost = [element[0] for element in model.log]

plt.plot(x, cost)
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.title('Cost function for each epoch')
plt.show()