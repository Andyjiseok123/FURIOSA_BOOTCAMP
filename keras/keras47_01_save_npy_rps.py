import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#1. 데이터
# file path
img_path = './_data/image/rps/'
img_datagen = ImageDataGenerator(
    rescale=1./255
)
img_data = img_datagen.flow_from_directory(
    img_path,
    target_size=(100,100),
    batch_size=5000,                      
    class_mode='categorical',            
    color_mode='rgb',
    shuffle=True,  
)
# print(type(img_data))                       #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(img_data[0][0].shape)                 #(1027, 100, 100, 3)
# print(img_data[0][1].shape)                 #(1027, 2)
# exit()
# print(img_data[0][0])                       
# print(img_data[0][1])

# exit()

x_train, x_test, y_train, y_test= train_test_split(img_data[0][0],img_data[0][1], 
                                                   train_size=0.85,random_state=4132, shuffle=True)

np_path = './_data/rps_npy/'
np.save(np_path+'keras47_01_x_train.npy', arr=x_train)
np.save(np_path+'keras47_01_y_train.npy', arr=y_train)
np.save(np_path+'keras47_01_x_test.npy', arr=x_test)
np.save(np_path+'keras47_01_y_test.npy', arr=y_test)
