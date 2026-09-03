#fetch_california_housing이 다운로드 되지 않을때
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
# print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
# print(y_train.shape, y_test.shape) #(404,) (102,)

#2. 모델 구성
model = Sequential()
model.add(Dense(78,input_dim=13))
model.add(Dense(39))
model.add(Dense(13))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=3000, batch_size=25)

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
