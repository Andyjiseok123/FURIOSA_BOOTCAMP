import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf
import random

random.seed(3212)
np.random.seed(3212)
tf.random.set_seed(3212)

from keras.applications import VGG16
from keras.datasets import cifar100

#2. 모델
vgg16 = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
)
vgg16.trainable = False
vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10,activation='softmax'))

model.summary()
"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ vgg16 (Functional)                   │ (None, 1, 1, 512)           │      14,714,688 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 512)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 100)                 │          51,300 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 100)                 │          10,100 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │           1,010 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 14,777,098 (56.37 MB)
 Trainable params: 62,410 (243.79 KB)
 Non-trainable params: 14,714,688 (56.13 MB)
"""

