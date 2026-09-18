#36-2 copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score


#1. 데이터
(x_train,y_train),(x_test, y_test) = mnist.load_data()
# print(x_test.shape)        #(10000,28,28)
# print(y_test.shape)        #(10000,)

print(np.max(x_train), np.min(x_train))     #255 0
print(np.max(x_test), np.min(x_test))       #255 0

########################### 스케일링 1. ###########################
x_train = x_train/255.
x_test = x_test/255.

x_train = x_train.reshape(-1,28*28)
x_test = x_test.reshape(-1,28*28)
# print(x_train.shape)            #(60000, 784)
# print(x_test.shape)             #(10000, 784)
# exit()
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
# model = Sequential()
# model.add(Dense(512,activation='relu', input_dim = 28*28))
# model.add(Dropout(0.4))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(10,activation='softmax'))

input1 = Input(shape=(28*28,))
dense1 = Dense(512,activation = 'relu',name='ys1')(input1)
drop1 = Dropout(0.4)(dense1)
dense2 = Dense(256,activation = 'relu', name= 'ys2')(drop1)
drop2 = Dropout(0.2)(dense2)
dense3 = Dense(128,activation = 'relu', name= 'ys3')(drop2)
drop3 = Dropout(0.2)(dense3)
dense4 = Dense(64,activation = 'relu', name= 'ys4')(drop3)
drop4 = Dropout(0.2)(dense4)
dense5 = Dense(16,activation = 'relu', name= 'ys5')(drop4)
output1 = Dense(10,activation='softmax')(dense5)

model = Model(inputs=input1, outputs=output1)


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

# accuraccy_score:  0.9905
# 소요시간 :  362.63 초


# accuraccy_score:  0.9938
# 소요시간 :  365.46 초

# DNN
# accuraccy_score:  0.9812
# 소요시간 :  210.9 초

# 함수형 모델
# accuraccy_score:  0.9823
# 소요시간 :  203.93 초