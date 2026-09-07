import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
# print(datasets)
# print(x.shape, y.shape) #(442,10) (442,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=3721)

#2. 모델 구성
model = Sequential()
model.add(Dense(16,input_dim=10))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=20, verbose = 1, validation_split = 0.33)
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
