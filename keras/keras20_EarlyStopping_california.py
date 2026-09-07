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

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=442)

#2. 모델 구성
model = Sequential()
model.add(Dense(10,activation='relu',input_dim=8))
model.add(Dense(7,activation='relu'))
model.add(Dense(9,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',       # validation loss 를 기준으로
    mode = 'min',               # 최솟값을 찾겠다.
    patience = 10,              # 10회동안 갱신 확인
    restore_best_weights= True, # False면 patience 이후값을 반환
)



start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=2000, 
                 batch_size=32,
                 validation_split = 0.33,
                 callbacks=[es],
                 ) #train set 약 15000개
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
plt.plot(np.log(hist.history['loss']), c='red', label='loss')
plt.plot(np.log(hist.history['val_loss']), c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('캘리포니아 Loss')
plt.xlabel('epoch')
plt.ylabel('los')
plt.grid()
plt.show()
