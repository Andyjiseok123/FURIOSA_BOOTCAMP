import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import time
from sklearn.metrics import accuracy_score

#1. 데이터

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    # horizontal_flip = True,              #수평반전
    # vertical_flip= True,                #수직반전
    # width_shift_range= 0.1,             #평형이동
    # height_shift_range= 0.1,            #수직이동
    # rotation_range=5,                   #각도조절
    # zoom_range=1.2,                     #확대축소
    # shear_range=0.7,                    #전단변형
    # fill_mode='nearest',                #채우기
    )

test_datagen = ImageDataGenerator(
    rescale=1./255,
    )

path_train = './_data/image/cat_dog/training_set/'
path_test = './_data/image/cat_dog/test_set/'

start_time = time.time()
xy_train = train_datagen.flow_from_directory(
    path_train,                 #경로
    target_size=(100,100),
    batch_size=10000,                      
    class_mode='binary',            #이진분류
    color_mode='rgb',
    shuffle=True,
    )

xy_test = test_datagen.flow_from_directory(
    path_test,
    target_size=(100,100),
    batch_size=10000,                      
    class_mode='binary',            #이진분류
    color_mode='rgb',
    shuffle=False,             # test에서는 필요가 없다
)

x_train = xy_train[0][0]
y_train = xy_train[0][1]

x_test = xy_test[0][0]
y_test = xy_test[0][1]

np_path = './_data/kaggle_cat_dog_npy/'
np.save(np_path+'keras45_01_x_train.npy', arr=x_train)
np.save(np_path+'keras45_01_y_train.npy', arr=y_train)
np.save(np_path+'keras45_01_x_test.npy', arr=x_test)
np.save(np_path+'keras45_01_y_test.npy', arr=y_test)

end_time = time.time()

print('변환 소요시간:', round(end_time-start_time,2),'초')


# 변환 소요시간: 35.03 초


"""
============================================================================================
이미지의 크기가 커지고 갯수가 많아질수록 학습시킬때 마다 이미지를 불러오는것보다
이미지를 numpy array형식으로 한번만 변환시켜놓고 불러오는것이 빠름
============================================================================================
"""