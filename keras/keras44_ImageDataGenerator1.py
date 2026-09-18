import numpy as np
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,              #수평반전
    vertical_flip= True,                #수직반전
    width_shift_range= 0.1,             #평형이동
    height_shift_range= 0.1,            #수직이동
    rotation_range=5,                   #각도조절
    zoom_range=1.2,                     #확대축소
    shear_range=0.7,                    #전단변형
    fill_mode='nearest',                #채우기
    )

test_datagen = ImageDataGenerator(
    rescale=1./255,
)

path_train = './_data/image/brain/train/'
path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(
    path_train,                 #경로
    target_size=(100,100),
    batch_size=10,                      
    class_mode='binary',            #이진분류
    color_mode='grayscale',
    shuffle=True,
    )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10,                      
    class_mode='binary',            #이진분류
    color_mode='grayscale',
    shuffle=False,             # test에서는 필요가 없다
)

# print(xy_train)                 #<keras.preprocessing.image.DirectoryIterator object at 0x0000019C0A097FA0>
# print(xy_train.next())

print(xy_train[0][0].shape)     #(10, 100, 100, 1)
print(xy_train[0][1].shape)     #(10,)


