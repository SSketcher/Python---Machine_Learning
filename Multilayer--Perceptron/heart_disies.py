from perceptron import Perceptron, Layer
import numpy as np
import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data')
data = df.iloc[:, :13].values
labels = df.iloc[:, 13:].values
labels = np.reshape(labels, (len(labels), 1))

data[data == '?'] = -1      #marking valeues '?' as -1

model = Perceptron([
    Layer(size = len(data[0])),
    Layer(size = 8),
    Layer(size = 8),
    Layer(size = 4)], activation = 'sigmoid')