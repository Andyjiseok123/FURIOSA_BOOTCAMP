#fetch_california_housing이 다운로드 되지 않을때
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import boston_housing
from sklearn.model_selection import train_test_split
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, x_test.shape) #(404, 13) (102, 13)
print(y_train.shape, y_test.shape) #(404,) (102,)

#2. 모델 구성
model = Sequential()
model.add(Dense(7,activation='relu',input_dim=13))
model.add(Dense(5,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(7,activation='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(
    monitor = 'val_loss',       # validation loss 를 기준으로
    mode = 'min',               # 최솟값을 찾겠다.
    patience = 50,              # 10회동안 갱신 확인
    restore_best_weights= True, # False면 patience 이후값을 반환
)

start_time = time.time()
hist = model.fit(x_train, y_train, 
                 epochs=5000, 
                 batch_size=100, 
                 verbose = 1, 
                 validation_split = 0.33,
                 callbacks=[es])
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
plt.plot(np.log(hist.history['loss'][10:]), c='red', label='loss')
plt.plot(np.log(hist.history['val_loss'][10:]), c='blue', label='val_loss')
plt.legend(loc='upper right')
plt.title('보스턴 Loss')
plt.xlabel('epoch')
plt.ylabel('los')
plt.grid()
plt.show()
