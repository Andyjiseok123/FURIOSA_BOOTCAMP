# R2 기준 0.55

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
print(x.shape)          #(20640, 8)
print(y.shape)          #(20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=333)

#2. 모델
model = Sequential()
model.add(Dense(24,input_dim = 8))
model.add(Dense(12))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 5000, batch_size = 500)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print("loss(mse) : " ,loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_predict)
print("Accuracy(r2) : ", r2)

mse = mean_squared_error(y_test, y_predict)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

# loss(mse) :  0.6207893490791321
# Accuracy(r2) :  0.5090853462583381
# RMSE :  0.7879018563777809