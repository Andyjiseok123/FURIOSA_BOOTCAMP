import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import VGG16

model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32,32,3),
)

# model.trainable=False

model.layers[8].trainable = False

print(len(model.weights))                       #30
print(len(model.trainable_weights))             #4

import pandas as pd
pd.set_option('max_colwidth',None)
layers=  [
    (type(layer).__name__, layer.name, layer.trainable)
    for layer in model.layers
]

results = pd.DataFrame(
    layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']
)

print(results)
