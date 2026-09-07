#fetch_california_housing이 다운로드 되지 않을때
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=442)
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, train_size=70/85, random_state=914)

#2. 모델 구성
model = Sequential()
model.add(Dense(5,activation='relu',input_dim=8))
model.add(Dense(7,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=2000, batch_size=500,
          verbose = 1, validation_data=(x_val,y_val)) #train set 약 15000개

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss : " ,loss)

"""
Hyperparameter tuning
file                :   None
remarks             :   Add activation function(Relu) at hidden layer
random_state        :   442,914
train_size          :   0.85
validation_size     :   0.15
test_size           :   0.15
list_percepticon    :   [5,7,9,7,3,1]
epochs              :   2000
batch_size          :   500
MSE value           :   1.5501
"""