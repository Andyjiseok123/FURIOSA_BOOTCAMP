#fetch_california_housing이 다운로드 되지 않을때
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# 11-1 copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (20640, 8) (20640,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, random_state=442)

#2. 모델 구성
model = Sequential()
model.add(Dense(5,input_dim=8))
model.add(Dense(7,activation='relu'))
model.add(Dense(9))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=500,
          validation_split = 0.2) #train set 약 15000개
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)

print("loss : " ,loss)

print("소요시간 : ", round(end_time-start_time), "초")

print("============================ hist =============================")
print(hist)
print("============================ hist.history =============================")
print(hist.history)
print("============================ loss =============================")
print(hist.history['loss'])
print("============================ val_loss =============================")
print(hist.history['val_loss'])

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'][3:], c='red', label='loss')
plt.plot(hist.history['val_loss'][3:], c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('캘리포니아 Loss')
plt.xlabel('epoch')
plt.ylabel('los')
plt.grid()
plt.show()
