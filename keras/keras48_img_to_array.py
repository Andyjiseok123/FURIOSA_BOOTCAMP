from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
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

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path+ 'keras48_me.npy',arr=arr)