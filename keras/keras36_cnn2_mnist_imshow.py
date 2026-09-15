import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D

(x_train,y_train),(x_test, y_test) = mnist.load_data()

# print(x_train.shape)        #(60000,28,28)
# print(y_train.shape)        #(60000,)

# print(x_test.shape)        #(10000,28,28)
# print(y_test.shape)        #(10000,)

# print(np.unique(y_train,return_counts=True))
# #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],dtype=int64))

plt.imshow(x_train[random.randrange(0,60001)],'gray')
plt.show()