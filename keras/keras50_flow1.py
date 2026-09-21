from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

path = './_data/image/'

img = load_img(path + 'my_image.png', target_size = (100,100))

# print(img)
# #<PIL.Image.Image image mode=RGB size=100x100 at 0x1F1CF5F5660>
# print(type(img))  #<class 'PIL.Image.Image'>

# plt.imshow(img)
# plt.show()

arr = img_to_array(img)
print(arr)
print(type(arr))    #<class 'numpy.ndarray'>
print(arr.shape)    #(100, 100, 3)

arr = np.expand_dims(arr,axis=0)        #차원증가

# np_path = './_data/kaggle_cat_dog_npy/'
# np.save(np_path+ 'keras48_me.npy',arr=arr)

datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    # height_shift_range=0.1,
    rotation_range=5,
    # zoom_range=1.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

it = datagen.flow(arr,batch_size=1,
                  )


print(it)
# print(it.next()) #python 3.10
print(next(it))
print(next(it).shape)           #(1, 100, 100, 3)


fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(5,5))
for i in range(5):
    batch = next(it)
    batch = batch.reshape(100,100,3)

    ax[i].imshow(batch)
    ax[i].axis('off')

plt.show()