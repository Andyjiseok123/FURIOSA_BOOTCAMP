# 50-1 copy
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=2,
    # zoom_range=1.1,
    # shear_range=0.7,
    # fill_mode='nearest'
)

augment_size = 100

print(x_train.shape)            #(60000,28,28)
print(x_train[0].shape)         #(28,28)

##################### 단순 복붙 ########################
aaa = np.tile(x_train[0], augment_size).reshape(-1,28,28,1)
print(aaa.shape)


xy_data = datagen.flow(
    np.tile(x_train[0].reshape(28,28),augment_size).reshape(-1,28,28,1),
    np.zeros(augment_size),
    batch_size=augment_size,
    shuffle=False,
).next()

# print(xy_data)
# print(type(xy_data))
# print(xy_data.shape)          # AttributeError: 'tuple' object has no attribute 'shape'
# print(len(xy_data))             # 2 : x,y

# print(xy_data[0].shape)         # (100, 28, 28, 1)
# print(xy_data[1].shape)         # (100,)

plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.imshow(xy_data[0][i], cmap='gray')

plt.show()