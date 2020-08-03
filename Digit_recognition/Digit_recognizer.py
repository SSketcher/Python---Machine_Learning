import numpy as np
import tensorflow as tf
import pandas as pd

#Training dataset
mnist = tf.keras.datasets.mnist
(training_data, training_labels), (test_data, test_labels) = mnist.load_data()
training_data, test_data = training_data / 255.0, test_data / 255.0


#Creating a model
