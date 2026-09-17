#36-2 copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import random


#1. 데이터
(x_train,y_train),(x_test, y_test) = fashion_mnist.load_data()
# plt.imshow(x_train[random.randrange(0,60001)],'gray')
# plt.show()
# exit()
# print(np.unique(y_train,return_counts=True))            #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],dtype=int64))

# print(x_test.shape)        #(10000,28,28)
# print(y_test.shape)        #(10000,)
# print(np.max(x_train), np.min(x_train))     #255 0
# print(np.max(x_test), np.min(x_test))       #255 0

########################### 스케일링 1. ###########################
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
# print(x_train.shape)            #(60000, 28, 28, 1)
# print(x_test.shape)             #(10000, 28, 28, 1)

# print(np.max(x_train), np.min(x_train))     #1.0 0.0
# print(np.max(x_test), np.min(x_test))       #1.0 0.0

########################### 스케일링 2. ###########################
# x_train = (x_train-127.5)/127.5
# x_test = (x_test-127.5)/127.5

# print(np.max(x_train), np.min(x_train))     #1.0 -1.0
# print(np.max(x_test), np.min(x_test))       #1.0 -1.0

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.fit_transform(y_test.reshape(-1,1))

#2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (3,3),padding='same', input_shape = (28,28,1)))                      #(26,26,64)
model.add(MaxPooling2D())
model.add(Conv2D(filters=32 , kernel_size=(3,3),padding='same', activation='relu'))        #(24,24,32)
model.add(Dropout(0.5))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))                               #(23,23,32)
model.add(MaxPooling2D())
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))                               #(22,22,16)
model.add(Dropout(0.4))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))                               #(21,21,16)
model.add(Dropout(0.2))
model.add(Conv2D(16,(3,3),padding='same',activation='relu'))                               #(20,20,16)

model.add(Flatten())                                                        #이후 FC layer와 붙기 위해 한줄로 reshape

model.add(Dense(64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(10,activation='softmax'))                                   

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
              metrics = ['acc'])
start_time = time.time()
es = EarlyStopping(monitor='val_loss',
                   mode='auto',
                   patience=50,
                   restore_best_weights=True)
model.fit(x_train,y_train,
          epochs = 2000, batch_size = 256,
          verbose = 1,
          validation_split = 0.3,
          callbacks = [es])
end_time = time.time()

#4. 평가, 예측
print("=======================model.evaluate===========================")
loss = model.evaluate(x_test,y_test,
                     verbose = 1)
print('loss: ',loss[0])
print('acc: ',loss[1])

y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis=1).reshape(-1,1)
y_test = np.argmax(y_test,axis=1).reshape(-1,1)

acc_score = accuracy_score(y_test,y_predict)
print('accuraccy_score: ',acc_score)
print('소요시간 : ',round(end_time-start_time,2), '초')

# accuraccy_score:  0.9222
# 소요시간 :  396.29 초