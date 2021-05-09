from perceptron import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

df = pd.read_csv('resources/iris.csv', header = 0)
labels = df.iloc[:, 4].values
data = df.iloc[:, 0:3].values
labels = np.reshape(labels, (len(labels), 1))

vocab = ['setosa', 'virginica', 'versicolor']
key_word = vocab[0]
labels = np.where(labels == key_word, 1, 0)
    
model = Perceptron(len(data[0]))
model.fit(data, np.array(labels), epochs=5)


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.set_title('Iris Dataset with the decision boundary')
ax.set_xlabel("Sepal length [cm]")
ax.set_ylabel("Sepal width [cm]")
ax.set_zlabel("Petal length [cm]")

d,a,b,c = model.weights

x_min = np.nanmin(np.array([row[0] for row in data]))
x_max = np.nanmax(np.array([row[0] for row in data]))

y_min = np.nanmin(np.array([row[1] for row in data]))
y_max = np.nanmax(np.array([row[1] for row in data]))

x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)

Xs,Ys = np.meshgrid(x,y)
Zs = (d - a*Xs - b*Ys) / c

ax.scatter(data[:50, 0], data[:50, 1], data[:50, 2], color='red',
           marker='o', s=4, edgecolor='red', label="Iris Setosa")
ax.scatter(data[50:100, 0], data[50:100, 1], data[50:100, 2], color='blue',
           marker='^', s=4, edgecolor='blue', label="Iris Versicolour")
ax.scatter(data[100:150, 0], data[100:150, 1],data[100:150, 2], color='green',
           marker='x', s=4, edgecolor='green', label="Iris Virginica")
ax.plot_surface(Xs, Ys, Zs, alpha=0.45)

plt.legend(loc='upper left')
plt.show()


result = model.predict(data[20])
print('Model predicted: ', result)
print('Truth is: ', labels[20])