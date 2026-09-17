import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

#2. 모델
model = Sequential()
model.add(Conv2D(10,(5,5),input_shape = (10,10,1),
                 padding='same',
                 strides=1
                 ))

model.add(Conv2D(filters=9, kernel_size=(3,3),
                 padding='valid',       #디폴트
                 strides=1              #디폴트
                 ))

model.summary()

"""
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 10, 10, 10)        260       
                                                                 
 conv2d_1 (Conv2D)           (None, 8, 8, 9)           819       
                                                                 
=================================================================
Total params: 1,079
Trainable params: 1,079
Non-trainable params: 0
_________________________________________________________________
"""