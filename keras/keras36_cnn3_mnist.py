#36-2 copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D


#1. 데이터
(x_train,y_train),(x_test, y_test) = mnist.load_data()
# print(x_test.shape)        #(10000,28,28)
# print(y_test.shape)        #(10000,)

print(np.max(x_train), np.min(x_train))     #255 0
print(np.max(x_test), np.min(x_test))       #255 0

########################### 스케일링 1. ###########################
# x_train = x_train/255.
# x_test = x_test/255.

# print(np.max(x_train), np.min(x_train))     #1.0 0.0
# print(np.max(x_test), np.min(x_test))       #1.0 0.0

########################### 스케일링 2. ###########################
x_train = (x_train-127.5)/127.5
x_test = (x_test-127.5)/127.5

print(np.max(x_train), np.min(x_train))     #1.0 0.0
print(np.max(x_test), np.min(x_test))       #1.0 0.0